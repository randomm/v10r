"""
Tests for v10r.database.introspection module.

Tests the database schema introspection utilities including
ColumnInfo, TableInfo, IndexInfo dataclasses and SchemaIntrospector class.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Optional, Any

from v10r.database.introspection import (
    ColumnType, ColumnInfo, IndexInfo, TableInfo, SchemaIntrospector
)
from v10r.database.connection import ConnectionPool
from v10r.exceptions import DatabaseError, SchemaError


class TestColumnType:
    """Test ColumnType enum."""
    
    def test_column_type_values(self):
        """Test ColumnType enum values."""
        assert ColumnType.TEXT == "text"
        assert ColumnType.VARCHAR == "varchar"
        assert ColumnType.VECTOR == "vector"
        assert ColumnType.INTEGER == "integer"
        assert ColumnType.BIGINT == "bigint"
        assert ColumnType.TIMESTAMP == "timestamp"
        assert ColumnType.TIMESTAMPTZ == "timestamptz"
        assert ColumnType.BOOLEAN == "boolean"
        assert ColumnType.JSON == "json"
        assert ColumnType.JSONB == "jsonb"
        assert ColumnType.UNKNOWN == "unknown"


class TestColumnInfo:
    """Test ColumnInfo dataclass."""
    
    def test_column_info_basic(self):
        """Test basic ColumnInfo creation."""
        col = ColumnInfo(
            name="test_col",
            data_type="text",
            is_nullable=True
        )
        
        assert col.name == "test_col"
        assert col.data_type == "text"
        assert col.is_nullable is True
        assert col.default_value is None
        assert col.max_length is None
    
    def test_column_info_with_defaults(self):
        """Test ColumnInfo with default values."""
        col = ColumnInfo(
            name="id",
            data_type="integer",
            is_nullable=False,
            default_value="nextval('seq'::regclass)"
        )
        
        assert col.name == "id"
        assert col.default_value == "nextval('seq'::regclass)"
        assert col.default == "nextval('seq'::regclass)"  # Should sync
    
    def test_column_info_default_sync(self):
        """Test that default and default_value are synced."""
        # Test setting default
        col = ColumnInfo(
            name="test",
            data_type="text",
            is_nullable=True,
            default="'test'"
        )
        assert col.default_value == "'test'"
        
        # Test setting default_value
        col2 = ColumnInfo(
            name="test2",
            data_type="text",
            is_nullable=True,
            default_value="'test2'"
        )
        assert col2.default == "'test2'"
    
    def test_column_type_property_text(self):
        """Test column_type property for text types."""
        col_text = ColumnInfo(name="col", data_type="text", is_nullable=True)
        assert col_text.column_type == ColumnType.TEXT
        
        col_varchar = ColumnInfo(name="col", data_type="character varying", is_nullable=True)
        assert col_varchar.column_type == ColumnType.VARCHAR
        
        col_varchar2 = ColumnInfo(name="col", data_type="varchar", is_nullable=True)
        assert col_varchar2.column_type == ColumnType.VARCHAR
    
    def test_column_type_property_vector(self):
        """Test column_type property for vector type."""
        col = ColumnInfo(
            name="embedding",
            data_type="vector",
            is_nullable=True,
            udt_name="vector"
        )
        assert col.column_type == ColumnType.VECTOR
    
    def test_column_type_property_numeric(self):
        """Test column_type property for numeric types."""
        col_int = ColumnInfo(name="col", data_type="integer", is_nullable=True)
        assert col_int.column_type == ColumnType.INTEGER
        
        col_int4 = ColumnInfo(name="col", data_type="int4", is_nullable=True)
        assert col_int4.column_type == ColumnType.INTEGER
        
        col_bigint = ColumnInfo(name="col", data_type="bigint", is_nullable=True)
        assert col_bigint.column_type == ColumnType.BIGINT
        
        col_int8 = ColumnInfo(name="col", data_type="int8", is_nullable=True)
        assert col_int8.column_type == ColumnType.BIGINT
    
    def test_column_type_property_timestamp(self):
        """Test column_type property for timestamp types."""
        col_ts = ColumnInfo(name="col", data_type="timestamp without time zone", is_nullable=True)
        assert col_ts.column_type == ColumnType.TIMESTAMP
        
        col_tstz = ColumnInfo(name="col", data_type="timestamp with time zone", is_nullable=True)
        assert col_tstz.column_type == ColumnType.TIMESTAMPTZ
    
    def test_column_type_property_other(self):
        """Test column_type property for other types."""
        col_bool = ColumnInfo(name="col", data_type="boolean", is_nullable=True)
        assert col_bool.column_type == ColumnType.BOOLEAN
        
        col_json = ColumnInfo(name="col", data_type="json", is_nullable=True)
        assert col_json.column_type == ColumnType.JSON
        
        col_jsonb = ColumnInfo(name="col", data_type="jsonb", is_nullable=True)
        assert col_jsonb.column_type == ColumnType.JSONB
        
        col_unknown = ColumnInfo(name="col", data_type="custom_type", is_nullable=True)
        assert col_unknown.column_type == ColumnType.UNKNOWN
    
    def test_is_vector_column(self):
        """Test is_vector_column property."""
        vector_col = ColumnInfo(name="emb", data_type="vector", is_nullable=True, udt_name="vector")
        assert vector_col.is_vector_column is True
        
        text_col = ColumnInfo(name="text", data_type="text", is_nullable=True)
        assert text_col.is_vector_column is False
    
    def test_is_text_column(self):
        """Test is_text_column property."""
        text_col = ColumnInfo(name="content", data_type="text", is_nullable=True)
        assert text_col.is_text_column is True
        
        varchar_col = ColumnInfo(name="title", data_type="varchar", is_nullable=True)
        assert varchar_col.is_text_column is True
        
        int_col = ColumnInfo(name="id", data_type="integer", is_nullable=False)
        assert int_col.is_text_column is False
    
    def test_column_info_str(self):
        """Test ColumnInfo string representation."""
        # Basic column
        col = ColumnInfo(name="title", data_type="text", is_nullable=True)
        assert str(col) == "title text"
        
        # Column with NOT NULL
        col_nn = ColumnInfo(name="id", data_type="integer", is_nullable=False)
        assert str(col_nn) == "id integer NOT NULL"
        
        # Column with default
        col_def = ColumnInfo(
            name="created_at", 
            data_type="timestamp", 
            is_nullable=False,
            default_value="now()"
        )
        assert str(col_def) == "created_at timestamp NOT NULL DEFAULT now()"
        
        # Vector column with dimension
        col_vec = ColumnInfo(
            name="embedding",
            data_type="vector",
            is_nullable=True,
            vector_dimension=1536
        )
        assert str(col_vec) == "embedding vector(1536)"
        
        # Varchar with max_length
        col_varchar = ColumnInfo(
            name="name",
            data_type="varchar",
            is_nullable=False,
            max_length=255
        )
        assert str(col_varchar) == "name varchar(255) NOT NULL"


class TestIndexInfo:
    """Test IndexInfo dataclass."""
    
    def test_index_info_basic(self):
        """Test basic IndexInfo creation."""
        idx = IndexInfo(
            name="idx_title",
            table_schema="public",
            table_name="documents",
            columns=["title"],
            is_unique=False,
            is_primary=False,
            index_type="btree",
            definition="CREATE INDEX idx_title ON public.documents (title)"
        )
        
        assert idx.name == "idx_title"
        assert idx.table_schema == "public"
        assert idx.table_name == "documents"
        assert idx.columns == ["title"]
        assert idx.is_unique is False
        assert idx.is_primary is False
        assert idx.index_type == "btree"
    
    def test_is_vector_index(self):
        """Test is_vector_index property."""
        # Vector index with ivfflat
        vector_idx = IndexInfo(
            name="idx_embedding",
            table_schema="public",
            table_name="docs",
            columns=["embedding"],
            is_unique=False,
            is_primary=False,
            index_type="ivfflat",
            definition="CREATE INDEX idx_embedding ON docs USING ivfflat (embedding)"
        )
        assert vector_idx.is_vector_index is True
        
        # Regular btree index
        btree_idx = IndexInfo(
            name="idx_title",
            table_schema="public",
            table_name="docs",
            columns=["title"],
            is_unique=False,
            is_primary=False,
            index_type="btree",
            definition="CREATE INDEX idx_title ON docs (title)"
        )
        assert btree_idx.is_vector_index is False
        
        # Vector index with "vector" in type
        vector_idx2 = IndexInfo(
            name="idx_vec",
            table_schema="public",
            table_name="docs",
            columns=["vec"],
            is_unique=False,
            is_primary=False,
            index_type="vector_ops",
            definition="CREATE INDEX idx_vec ON docs (vec)"
        )
        assert vector_idx2.is_vector_index is True


class TestTableInfo:
    """Test TableInfo dataclass."""
    
    def test_table_info_basic(self):
        """Test basic TableInfo creation."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "title": ColumnInfo(name="title", data_type="text", is_nullable=True)
        }
        
        table = TableInfo(
            schema="public",
            name="documents",
            columns=columns,
            indexes=[]
        )
        
        assert table.schema == "public"
        assert table.name == "documents"
        assert len(table.columns) == 2
        assert table.primary_key == []  # Should initialize empty list
    
    def test_table_info_with_primary_key(self):
        """Test TableInfo with primary key."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "title": ColumnInfo(name="title", data_type="text", is_nullable=True)
        }
        
        table = TableInfo(
            schema="public",
            name="documents",
            columns=columns,
            indexes=[],
            primary_key=["id"]
        )
        
        assert table.primary_key == ["id"]
    
    def test_full_name_property(self):
        """Test full_name property."""
        table = TableInfo(
            schema="content",
            name="articles",
            columns={},
            indexes=[]
        )
        
        assert table.full_name == "content.articles"
    
    def test_vector_columns_property(self):
        """Test vector_columns property."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "content": ColumnInfo(name="content", data_type="text", is_nullable=True),
            "embedding": ColumnInfo(name="embedding", data_type="vector", is_nullable=True, udt_name="vector"),
            "embedding2": ColumnInfo(name="embedding2", data_type="vector", is_nullable=True, udt_name="vector")
        }
        
        table = TableInfo(
            schema="public",
            name="docs",
            columns=columns,
            indexes=[]
        )
        
        vector_cols = table.vector_columns
        assert len(vector_cols) == 2
        assert "embedding" in vector_cols
        assert "embedding2" in vector_cols
        assert "id" not in vector_cols
        assert "content" not in vector_cols
    
    def test_text_columns_property(self):
        """Test text_columns property."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "title": ColumnInfo(name="title", data_type="text", is_nullable=True),
            "description": ColumnInfo(name="description", data_type="varchar", is_nullable=True),
            "embedding": ColumnInfo(name="embedding", data_type="vector", is_nullable=True, udt_name="vector")
        }
        
        table = TableInfo(
            schema="public",
            name="docs",
            columns=columns,
            indexes=[]
        )
        
        text_cols = table.text_columns
        assert len(text_cols) == 2
        assert "title" in text_cols
        assert "description" in text_cols
        assert "id" not in text_cols
        assert "embedding" not in text_cols
    
    def test_has_column(self):
        """Test has_column method."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "title": ColumnInfo(name="title", data_type="text", is_nullable=True)
        }
        
        table = TableInfo(
            schema="public",
            name="docs",
            columns=columns,
            indexes=[]
        )
        
        assert table.has_column("id") is True
        assert table.has_column("title") is True
        assert table.has_column("nonexistent") is False
    
    def test_get_column(self):
        """Test get_column method."""
        id_col = ColumnInfo(name="id", data_type="integer", is_nullable=False)
        title_col = ColumnInfo(name="title", data_type="text", is_nullable=True)
        
        columns = {"id": id_col, "title": title_col}
        
        table = TableInfo(
            schema="public",
            name="docs",
            columns=columns,
            indexes=[]
        )
        
        assert table.get_column("id") == id_col
        assert table.get_column("title") == title_col
        assert table.get_column("nonexistent") is None
    
    def test_get_vector_dimension(self):
        """Test get_vector_dimension method."""
        columns = {
            "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
            "embedding": ColumnInfo(
                name="embedding", 
                data_type="vector", 
                is_nullable=True, 
                udt_name="vector",
                vector_dimension=1536
            ),
            "text_col": ColumnInfo(name="text_col", data_type="text", is_nullable=True)
        }
        
        table = TableInfo(
            schema="public",
            name="docs",
            columns=columns,
            indexes=[]
        )
        
        assert table.get_vector_dimension("embedding") == 1536
        assert table.get_vector_dimension("text_col") is None
        assert table.get_vector_dimension("nonexistent") is None


class TestSchemaIntrospector:
    """Test SchemaIntrospector class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool."""
        return MagicMock(spec=ConnectionPool)
    
    @pytest.fixture
    def introspector(self, mock_pool):
        """SchemaIntrospector instance with mocked pool."""
        return SchemaIntrospector(mock_pool)
    
    @pytest.mark.asyncio
    async def test_table_exists_true(self, introspector, mock_pool):
        """Test table_exists when table exists."""
        mock_pool.fetchval.return_value = True
        
        result = await introspector.table_exists("public", "documents")
        
        assert result is True
        mock_pool.fetchval.assert_called_once()
        # Check that the query looks for the table
        call_args = mock_pool.fetchval.call_args[0]
        assert "information_schema.tables" in call_args[0]
        assert call_args[1] == "public"
        assert call_args[2] == "documents"
    
    @pytest.mark.asyncio
    async def test_table_exists_false(self, introspector, mock_pool):
        """Test table_exists when table doesn't exist."""
        mock_pool.fetchval.return_value = False
        
        result = await introspector.table_exists("public", "nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_table_exists_exception(self, introspector, mock_pool):
        """Test table_exists with database exception."""
        mock_pool.fetchval.side_effect = Exception("Connection error")
        
        with pytest.raises(DatabaseError) as exc_info:
            await introspector.table_exists("public", "documents")
        
        assert "Failed to check table existence" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_columns_success(self, introspector, mock_pool):
        """Test get_columns with successful result."""
        # Mock column data
        mock_pool.fetch.return_value = [
            {
                'column_name': 'id',
                'data_type': 'integer',
                'is_nullable': 'NO',
                'column_default': 'nextval(\'seq\'::regclass)',
                'character_maximum_length': None,
                'numeric_precision': 32,
                'numeric_scale': 0,
                'ordinal_position': 1,
                'udt_name': 'int4',
                'is_generated': 'NEVER',
                'generation_expression': None
            },
            {
                'column_name': 'title',
                'data_type': 'text',
                'is_nullable': 'YES',
                'column_default': None,
                'character_maximum_length': None,
                'numeric_precision': None,
                'numeric_scale': None,
                'ordinal_position': 2,
                'udt_name': 'text',
                'is_generated': 'NEVER',
                'generation_expression': None
            }
        ]
        
        result = await introspector.get_columns("public", "documents")
        
        assert len(result) == 2
        assert "id" in result
        assert "title" in result
        
        id_col = result["id"]
        assert id_col.name == "id"
        assert id_col.data_type == "integer"
        assert id_col.is_nullable is False
        assert id_col.default_value == "nextval('seq'::regclass)"
        
        title_col = result["title"]
        assert title_col.name == "title"
        assert title_col.data_type == "text"
        assert title_col.is_nullable is True
        assert title_col.default_value is None
    
    @pytest.mark.asyncio
    async def test_get_columns_with_vector_dimension(self, introspector, mock_pool):
        """Test get_columns with vector column dimension lookup."""
        # Mock column data with vector column
        mock_pool.fetch.return_value = [
            {
                'column_name': 'embedding',
                'data_type': 'USER-DEFINED',
                'is_nullable': 'YES',
                'column_default': None,
                'character_maximum_length': None,
                'numeric_precision': None,
                'numeric_scale': None,
                'ordinal_position': 1,
                'udt_name': 'vector',
                'is_generated': 'NEVER',
                'generation_expression': None
            }
        ]
        
        # Mock vector dimension query
        with patch.object(introspector, 'get_vector_dimension', return_value=1536) as mock_get_dim:
            result = await introspector.get_columns("public", "documents")
            
            assert len(result) == 1
            embedding_col = result["embedding"]
            assert embedding_col.udt_name == "vector"
            assert embedding_col.vector_dimension == 1536
            mock_get_dim.assert_called_once_with("public", "documents", "embedding")
    
    @pytest.mark.asyncio
    async def test_get_vector_dimension_success(self, introspector, mock_pool):
        """Test get_vector_dimension with successful result."""
        mock_pool.fetchval.return_value = 1540  # 1536 + 4 (VARHDRSZ)
        
        result = await introspector.get_vector_dimension("public", "documents", "embedding")
        
        assert result == 1536  # Should subtract 4
        mock_pool.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_vector_dimension_not_found(self, introspector, mock_pool):
        """Test get_vector_dimension when dimension not found."""
        mock_pool.fetchval.return_value = None
        
        result = await introspector.get_vector_dimension("public", "documents", "embedding")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_table_info_success(self, introspector):
        """Test get_table_info with successful result."""
        # Mock the individual methods
        with patch.object(introspector, 'table_exists', return_value=True) as mock_exists:
            with patch.object(introspector, 'get_columns', return_value={
                "id": ColumnInfo(name="id", data_type="integer", is_nullable=False)
            }) as mock_columns:
                with patch.object(introspector, 'get_indexes', return_value=[]) as mock_indexes:
                    with patch.object(introspector, 'get_table_metadata', return_value={
                        "primary_key": ["id"],
                        "row_count": 100,
                        "table_type": "BASE TABLE",
                        "owner": "postgres"
                    }) as mock_metadata:
                        
                        result = await introspector.get_table_info("public", "documents")
                        
                        assert result is not None
                        assert result.schema == "public"
                        assert result.name == "documents"
                        assert len(result.columns) == 1
                        assert result.primary_key == ["id"]
                        assert result.row_count == 100
                        
                        mock_exists.assert_called_once_with("public", "documents")
                        mock_columns.assert_called_once_with("public", "documents")
                        mock_indexes.assert_called_once_with("public", "documents")
                        mock_metadata.assert_called_once_with("public", "documents")
    
    @pytest.mark.asyncio
    async def test_get_table_info_not_exists(self, introspector):
        """Test get_table_info when table doesn't exist."""
        with patch.object(introspector, 'table_exists', return_value=False):
            result = await introspector.get_table_info("public", "nonexistent")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_list_schemas_success(self, introspector, mock_pool):
        """Test list_schemas with successful result."""
        mock_pool.fetch.return_value = [
            {"schema_name": "public"},
            {"schema_name": "content"},
            {"schema_name": "v10r"}
        ]
        
        result = await introspector.list_schemas()
        
        assert result == ["public", "content", "v10r"]
        mock_pool.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_installed(self, introspector, mock_pool):
        """Test check_pgvector_extension when extension is installed."""
        mock_pool.fetchrow.return_value = {
            "installed_version": "0.5.0",
            "default_version": "0.5.0",
            "comment": "vector data type and ivfflat access method"
        }
        
        result = await introspector.check_pgvector_extension()
        
        assert result["available"] is True
        assert result["installed"] is True
        assert result["installed_version"] == "0.5.0"
        assert result["default_version"] == "0.5.0"
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_not_installed(self, introspector, mock_pool):
        """Test check_pgvector_extension when extension is not installed."""
        mock_pool.fetchrow.return_value = None
        
        result = await introspector.check_pgvector_extension()
        
        assert result["available"] is False
        assert result["installed"] is False 