"""
Column collision detection and resolution for v10r.

Handles detection of column naming conflicts and provides
strategies for safe resolution with comprehensive analysis.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..database.introspection import SchemaIntrospector, ColumnInfo, TableInfo
from ..database.connection import ConnectionPool
from ..exceptions import ColumnCollisionError, SchemaError


logger = logging.getLogger(__name__)


class CollisionSeverity(str, Enum):
    """Severity levels for column collisions."""
    
    SAFE = "safe"             # No collision (value: 1)
    INFO = "info"             # Unrelated column (value: 2)  
    WARNING = "warning"        # Likely vectorizer-related pattern (value: 3)
    CRITICAL = "critical"      # Existing vector column or previous v10r management (value: 4)
    
    def __lt__(self, other):
        """Enable comparison of severity levels."""
        if self.__class__ is other.__class__:
            order = {"safe": 1, "info": 2, "warning": 3, "critical": 4}
            return order[self.value] < order[other.value]
        return NotImplemented
    
    @property
    def order_value(self) -> int:
        """Get numeric order value for comparison."""
        order = {"safe": 1, "info": 2, "warning": 3, "critical": 4}
        return order[self.value]


class CollisionType(str, Enum):
    """Types of column collisions."""
    
    EXISTING_VECTOR = "existing_vector"               # Column is already a vector column
    PREVIOUS_VECTORIZER = "previous_vectorizer"       # Column was managed by v10r before
    VECTORIZER_PATTERN = "vectorizer_pattern"        # Name suggests vectorizer origin
    UNRELATED_COLUMN = "unrelated_column"             # Column exists but unrelated
    NO_COLLISION = "no_collision"                     # No collision detected


@dataclass
class CollisionInfo:
    """Information about a column collision."""
    
    schema_name: str  # Keep test-expected parameter name
    table_name: str
    desired_column: str
    existing_column_type: Optional[str]
    collision_type: CollisionType
    severity: CollisionSeverity
    details: str
    alternative_names: List[str] = None
    
    @property
    def table_schema(self) -> str:
        """Alias for schema_name for backward compatibility."""
        return self.schema_name
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical collision."""
        return self.severity == CollisionSeverity.CRITICAL
    
    @property 
    def is_safe(self) -> bool:
        """Check if this is a safe collision (no collision)."""
        return self.severity == CollisionSeverity.SAFE
    
    @property
    def needs_resolution(self) -> bool:
        """Check if this collision needs resolution."""
        return not self.is_safe
    
    @property
    def table_full_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.schema_name}.{self.table_name}"
    
    def __post_init__(self):
        if self.alternative_names is None:
            self.alternative_names = []


class ColumnCollisionDetector:
    """Detects and analyzes column naming collisions."""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.introspector = SchemaIntrospector(pool)
        
        # Patterns that suggest vectorizer management
        self.vectorizer_patterns = [
            r".*_vector$",
            r".*_embedding$", 
            r".*_cleaned$",
            r".*_model$",
            r".*_created_at$",
            r".*_updated_at$",
            r".*_vector_v\d+$",
            r".*_embedding_v\d+$",
        ]
    
    async def check_single_collision(
        self,
        schema: str,
        table: str,
        desired_column: str,
    ) -> CollisionInfo:
        """
        Check for collision on a single column (alias for detect_collision).
        """
        return await self.detect_collision(schema, table, desired_column)
    
    async def detect_collision(
        self,
        schema: str,
        table: str,
        desired_column: str,
        check_metadata: bool = True,
    ) -> CollisionInfo:
        """
        Detect and analyze a potential column collision.
        
        Args:
            schema: Target schema name
            table: Target table name
            desired_column: Desired column name
            check_metadata: Whether to check v10r metadata tables
            
        Returns:
            CollisionInfo with analysis results
        """
        try:
            # Get table information
            table_info = await self.introspector.get_table_info(schema, table)
            if not table_info:
                raise SchemaError(f"Table {schema}.{table} does not exist")
            
            # Check if column exists
            existing_column = table_info.get_column(desired_column)
            if not existing_column:
                return CollisionInfo(
                    schema_name=schema,
                    table_name=table,
                    desired_column=desired_column,
                    existing_column_type=None,
                    severity=CollisionSeverity.SAFE,
                    collision_type=CollisionType.NO_COLLISION,
                    details="No collision detected",
                )
            
            # Analyze the collision
            analysis = await self._analyze_collision(
                schema, table, desired_column, existing_column, check_metadata
            )
            
            # Generate alternative names if needed
            alternatives = []
            if analysis["severity"] in (CollisionSeverity.CRITICAL, CollisionSeverity.WARNING):
                existing_columns = {col.name for col in table_info.columns.values()}
                alternatives = self._generate_alternative_names(
                    desired_column, existing_columns
                )
            
            return CollisionInfo(
                schema_name=schema,
                table_name=table,
                desired_column=desired_column,
                existing_column_type=existing_column.data_type if existing_column else None,
                severity=analysis["severity"],
                collision_type=analysis["collision_type"], 
                details=analysis["details"].get("message", str(analysis["details"])),
                alternative_names=alternatives,
            )
            
        except Exception as e:
            logger.error(f"Error detecting collision for {schema}.{table}.{desired_column}: {e}")
            raise SchemaError(f"Collision detection failed: {e}") from e
    
    async def _analyze_collision(
        self,
        schema: str,
        table: str,
        column_name: str,
        existing_column: ColumnInfo,
        check_metadata: bool,
    ) -> Dict[str, Any]:
        """Analyze the nature and severity of a collision."""
        
        # Check if it's a vector column
        if existing_column.is_vector_column:
            return {
                "severity": CollisionSeverity.CRITICAL,
                "collision_type": CollisionType.EXISTING_VECTOR,
                "details": {
                    "data_type": existing_column.data_type,
                    "vector_dimension": existing_column.vector_dimension,
                    "message": "Column is already a pgvector column",
                    "risk": "High - may indicate duplicate configuration or manual vector column",
                },
                "suggested_resolution": "Check for existing v10r configuration or use different column name",
            }
        
        # Check metadata if requested and available
        if check_metadata:
            metadata_result = await self._check_metadata_history(schema, table, column_name)
            if metadata_result:
                return {
                    "severity": CollisionSeverity.CRITICAL,
                    "collision_type": CollisionType.PREVIOUS_VECTORIZER,
                    "details": {
                        "metadata_info": metadata_result,
                        "message": "Column was previously managed by v10r",
                        "risk": "High - may cause configuration conflicts",
                    },
                    "suggested_resolution": "Review metadata and existing configuration",
                }
        
        # Check for vectorizer naming patterns
        if self._matches_vectorizer_pattern(column_name):
            return {
                "severity": CollisionSeverity.WARNING,
                "collision_type": CollisionType.VECTORIZER_PATTERN,
                "details": {
                    "data_type": existing_column.data_type,
                    "pattern_matched": self._get_matched_pattern(column_name),
                    "message": "Column name suggests vectorizer origin",
                    "risk": "Medium - may indicate related vectorization activity",
                },
                "suggested_resolution": "Use alternative name or investigate column purpose",
            }
        
        # Unrelated column
        return {
            "severity": CollisionSeverity.INFO,
            "collision_type": CollisionType.UNRELATED_COLUMN,
            "details": {
                "data_type": existing_column.data_type,
                "max_length": existing_column.max_length,
                "is_nullable": existing_column.is_nullable,
                "message": "Column exists but appears unrelated to vectorization",
                "risk": "Low - likely safe to use alternative name",
            },
            "suggested_resolution": "Use alternative name with suffix",
        }
    
    async def _check_metadata_history(
        self, schema: str, table: str, column: str
    ) -> bool:
        """Check v10r metadata for column history."""
        try:
            # Check if v10r_metadata schema exists
            schemas = await self.introspector.list_schemas()
            if "v10r_metadata" not in schemas:
                return False
            
            # Check column registry
            query = """
                SELECT 
                    original_column_name,
                    actual_column_name,
                    column_type,
                    dimension,
                    model_name,
                    created_at,
                    deprecated_at,
                    config_key
                FROM v10r_metadata.column_registry
                WHERE database_name = current_database()
                AND schema_name = $1
                AND table_name = $2
                AND (original_column_name = $3 OR actual_column_name = $3)
                ORDER BY created_at DESC
                LIMIT 5
            """
            
            records = await self.pool.fetch(query, schema, table, column)
            if records:
                return True
            
            # Check collision log
            query = """
                SELECT 
                    desired_column_name,
                    collision_severity,
                    collision_type,
                    resolved_column_name,
                    occurred_at,
                    details
                FROM v10r_metadata.collision_log
                WHERE database_name = current_database()
                AND schema_name = $1
                AND table_name = $2
                AND desired_column_name = $3
                ORDER BY occurred_at DESC
                LIMIT 3
            """
            
            collision_records = await self.pool.fetch(query, schema, table, column)
            if collision_records:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Could not check metadata history for {schema}.{table}.{column}: {e}")
            return False
    
    def _matches_vectorizer_pattern(self, column_name: str) -> bool:
        """Check if column name matches vectorizer patterns."""
        return any(re.match(pattern, column_name, re.IGNORECASE) for pattern in self.vectorizer_patterns)
    
    def _get_matched_pattern(self, column_name: str) -> Optional[str]:
        """Get the pattern that matched the column name."""
        for pattern in self.vectorizer_patterns:
            if re.match(pattern, column_name, re.IGNORECASE):
                return pattern
        return None
    
    def _is_vector_column(self, column: ColumnInfo) -> bool:
        """Check if column is a pgvector column."""
        return (
            column.data_type == "USER-DEFINED" and 
            getattr(column, "udt_name", "") == "vector"
        ) or column.is_vector_column
    
    def _assess_collision_severity(
        self, 
        collision_type: CollisionType, 
        data_type: Optional[str], 
        has_metadata_history: bool
    ) -> CollisionSeverity:
        """Assess the severity of a collision."""
        if collision_type == CollisionType.EXISTING_VECTOR:
            return CollisionSeverity.CRITICAL
        elif collision_type == CollisionType.PREVIOUS_VECTORIZER:
            return CollisionSeverity.CRITICAL
        elif collision_type == CollisionType.VECTORIZER_PATTERN:
            return CollisionSeverity.WARNING if not has_metadata_history else CollisionSeverity.CRITICAL
        elif collision_type == CollisionType.UNRELATED_COLUMN:
            return CollisionSeverity.INFO
        else:  # NO_COLLISION
            return CollisionSeverity.SAFE
    
    def _generate_alternative_names(
        self, desired_name: str, existing_columns: set, strategy: Optional[str] = None
    ) -> List[str]:
        """Generate alternative column names."""
        alternatives = []
        base_name = desired_name
        
        # If specific strategy requested, use it
        if strategy == "prefix":
            # Add prefix-based alternatives
            prefixes = ["v10r_", "new_", "alt_", "v2_"]
            for prefix in prefixes:
                candidate = f"{prefix}{base_name}"
                if candidate not in existing_columns:
                    alternatives.append(candidate)
                    if len(alternatives) >= 5:
                        break
            return alternatives
        
        # Default strategy: suffix-based
        
        # Strategy 1: Add version suffix (v2, v3, etc.)
        for i in range(2, 10):
            candidate = f"{base_name}_v{i}"
            if candidate not in existing_columns:
                alternatives.append(candidate)
                if len(alternatives) >= 3:
                    break
        
        # Strategy 2: Add numeric suffix (2, 3, etc.)  
        for i in range(2, 10):
            candidate = f"{base_name}_{i}"
            if candidate not in existing_columns and candidate not in alternatives:
                alternatives.append(candidate)
                if len(alternatives) >= 5:
                    break
        
        # Strategy 3: Add descriptive suffixes
        descriptive_suffixes = ["_new", "_alt", "_updated", "_current"]
        for suffix in descriptive_suffixes:
            candidate = f"{base_name}{suffix}"
            if candidate not in existing_columns and candidate not in alternatives:
                alternatives.append(candidate)
                if len(alternatives) >= 7:
                    break
        
        return alternatives[:7]  # Return max 7 alternatives
    
    async def check_multiple_collisions(
        self,
        schema: str,
        table: str,
        desired_columns: List[str],
    ) -> Dict[str, CollisionInfo]:
        """Check for collisions across multiple columns."""
        results = {}
        
        for column in desired_columns:
            try:
                collision_info = await self.detect_collision(schema, table, column)
                results[column] = collision_info
            except Exception as e:
                logger.error(f"Error checking collision for {column}: {e}")
                results[column] = CollisionInfo(
                    schema_name=schema,
                    table_name=table,
                    desired_column=column,
                    existing_column_type=None,
                    severity=CollisionSeverity.CRITICAL,
                    collision_type=CollisionType.UNRELATED_COLUMN,
                    details=f"Error: {str(e)}",
                )
        
        return results
    
    async def get_collision_summary(self, collisions: Dict[str, CollisionInfo]) -> Dict[str, Any]:
        """Get a summary of collision analysis results."""
        total = len(collisions)
        by_severity = {}
        by_type = {}
        critical_collisions = []
        
        for collision in collisions.values():
            # Count by severity
            severity = collision.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by type
            col_type = collision.collision_type.value
            by_type[col_type] = by_type.get(col_type, 0) + 1
            
            # Track critical collisions
            if collision.is_critical:
                critical_collisions.append({
                    "column": collision.desired_column,
                    "type": collision.collision_type.value,
                    "details": collision.details.get("message", ""),
                })
        
        return {
            "total_columns_checked": total,
            "collisions_by_severity": by_severity,
            "collisions_by_type": by_type,
            "critical_collisions": critical_collisions,
            "has_critical_issues": len(critical_collisions) > 0,
            "safe_to_proceed": len(critical_collisions) == 0,
        } 