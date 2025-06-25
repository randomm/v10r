"""
Schema management package for v10r.

This package provides:
- Schema reconciliation core logic
- Column collision detection and resolution
- Safe ALTER TABLE operations with rollback
- Metadata tracking tables setup
"""

from .reconciler import SchemaReconciler, ReconciliationResult
from .collision_detector import ColumnCollisionDetector, CollisionInfo, CollisionSeverity
from .operations import SafeSchemaOperations, SchemaChange, ChangeType
from .metadata import MetadataManager, MetadataSchema

__all__ = [
    "SchemaReconciler",
    "ReconciliationResult", 
    "ColumnCollisionDetector",
    "CollisionInfo",
    "CollisionSeverity",
    "SafeSchemaOperations",
    "SchemaChange",
    "ChangeType",
    "MetadataManager",
    "MetadataSchema",
] 