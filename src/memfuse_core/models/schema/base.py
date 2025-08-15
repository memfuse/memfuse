"""Base schema class for database table definitions."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ColumnDefinition:
    """Definition of a database column."""
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    primary_key: bool = False
    unique: bool = False
    check_constraint: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class IndexDefinition:
    """Definition of a database index."""
    name: str
    columns: List[str]
    index_type: str = "btree"  # btree, gin, gist, hash, diskann, etc.
    unique: bool = False
    where_clause: Optional[str] = None
    with_options: Optional[Dict[str, Any]] = None
    comment: Optional[str] = None


@dataclass
class TriggerDefinition:
    """Definition of a database trigger."""
    name: str
    timing: str  # BEFORE, AFTER, INSTEAD OF
    event: str   # INSERT, UPDATE, DELETE
    function_name: str
    for_each: str = "ROW"  # ROW, STATEMENT
    when_condition: Optional[str] = None


class BaseSchema(ABC):
    """Base class for database table schemas."""
    
    def __init__(self):
        self.table_name = self.get_table_name()
        self.columns = self.define_columns()
        self.indexes = self.define_indexes()
        self.triggers = self.define_triggers()
        self.constraints = self.define_constraints()
    
    @abstractmethod
    def get_table_name(self) -> str:
        """Return the table name."""
        pass
    
    @abstractmethod
    def define_columns(self) -> List[ColumnDefinition]:
        """Define table columns."""
        pass
    
    def define_indexes(self) -> List[IndexDefinition]:
        """Define table indexes. Override in subclasses if needed."""
        return []
    
    def define_triggers(self) -> List[TriggerDefinition]:
        """Define table triggers. Override in subclasses if needed."""
        return []
    
    def define_constraints(self) -> List[str]:
        """Define table constraints. Override in subclasses if needed."""
        return []
    
    def generate_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement."""
        sql_parts = [f"CREATE TABLE IF NOT EXISTS {self.table_name} ("]
        
        # Add columns
        column_definitions = []
        for col in self.columns:
            col_def = f"    {col.name} {col.data_type}"
            
            if col.primary_key:
                col_def += " PRIMARY KEY"
            elif not col.nullable:
                col_def += " NOT NULL"
            
            if col.default is not None:
                col_def += f" DEFAULT {col.default}"
            
            if col.unique and not col.primary_key:
                col_def += " UNIQUE"
            
            if col.check_constraint:
                col_def += f" CHECK ({col.check_constraint})"
            
            column_definitions.append(col_def)
        
        sql_parts.append(",\n".join(column_definitions))
        
        # Add table constraints
        if self.constraints:
            sql_parts.append(",\n    " + ",\n    ".join(self.constraints))
        
        sql_parts.append(");")
        
        return "\n".join(sql_parts)
    
    def generate_indexes_sql(self) -> List[str]:
        """Generate CREATE INDEX SQL statements."""
        index_sqls = []
        
        for idx in self.indexes:
            sql = f"CREATE"
            
            if idx.unique:
                sql += " UNIQUE"
            
            sql += f" INDEX IF NOT EXISTS {idx.name}"
            sql += f" ON {self.table_name}"
            
            if idx.index_type != "btree":
                sql += f" USING {idx.index_type}"
            
            sql += f" ({', '.join(idx.columns)})"
            
            if idx.with_options:
                options = []
                for key, value in idx.with_options.items():
                    if isinstance(value, str):
                        options.append(f"{key} = '{value}'")
                    else:
                        options.append(f"{key} = {value}")
                sql += f" WITH ({', '.join(options)})"
            
            if idx.where_clause:
                sql += f" WHERE {idx.where_clause}"
            
            sql += ";"
            index_sqls.append(sql)
        
        return index_sqls
    
    def generate_triggers_sql(self) -> List[str]:
        """Generate trigger SQL statements."""
        trigger_sqls = []
        
        for trigger in self.triggers:
            # Drop existing trigger
            drop_sql = f"DROP TRIGGER IF EXISTS {trigger.name} ON {self.table_name};"
            trigger_sqls.append(drop_sql)
            
            # Create trigger
            sql = f"CREATE TRIGGER {trigger.name}"
            sql += f" {trigger.timing} {trigger.event}"
            sql += f" ON {self.table_name}"
            sql += f" FOR EACH {trigger.for_each}"
            
            if trigger.when_condition:
                sql += f" WHEN ({trigger.when_condition})"
            
            sql += f" EXECUTE FUNCTION {trigger.function_name}();"
            trigger_sqls.append(sql)
        
        return trigger_sqls
    
    def generate_all_sql(self) -> str:
        """Generate complete SQL for table creation."""
        sql_parts = []
        
        # Table creation
        sql_parts.append(self.generate_create_table_sql())
        sql_parts.append("")
        
        # Indexes
        index_sqls = self.generate_indexes_sql()
        if index_sqls:
            sql_parts.extend(index_sqls)
            sql_parts.append("")
        
        # Triggers
        trigger_sqls = self.generate_triggers_sql()
        if trigger_sqls:
            sql_parts.extend(trigger_sqls)
            sql_parts.append("")
        
        return "\n".join(sql_parts)
