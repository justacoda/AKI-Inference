# Relevant imports
from contextlib import contextmanager
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import threading

SQLInput = Union[
    str,
    Tuple[str, Sequence[Any]],
    Dict[str, Any]
]

class SQLite_Manager:
    def __init__(self, db_filepath:str="/db", timeout: float = 30.0):
        """
        Function containing all CRUD operations for SQLite
        """
        self.db_filepath = db_filepath
        self.timeout = timeout
        self._lock = threading.Lock()
        self._conn = None
        self._init_lock = threading.Lock()

    def _initialise_database(self):
        '''
        Create/connect to a database file under /db
        '''
        with self._init_lock:
            # If user passes a directory, pick a default file in it
            if os.path.isdir(self.db_filepath) or self.db_filepath.endswith(os.sep):
                self.db_filepath = os.path.join(self.db_filepath.rstrip(os.sep), "app.sqlite")

            # Ensure parent directory exists
            parent = os.path.dirname(self.db_filepath) or "."
            os.makedirs(parent, exist_ok=True)

        # Establish connection for this thread
        conn = sqlite3.connect(self.db_filepath, timeout = self.timeout, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # SQLite Pragmas
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.commit()

        # Ensure connection is locked to a thread
        self._conn = conn
        return conn

    # Ensure each thread access the database properly
    @property
    def _get_conn(self):
        # Initialise connection 
        if self._conn is None:
            self._conn = self._initialise_database()
        return self._conn

    @contextmanager
    def _cursor(self):
        '''
        Creates a cursor to run SQL commands and read results from SQLite connection
        '''
        with self._lock:
            cur = self._get_conn.cursor()
            try:
                yield cur
                self._get_conn.commit()
            except Exception:
                self._get_conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        '''
        Close connection to database
        '''
        # Thread specific
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -------------------------
    # Low-level execute helpers with cursor
    # -------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        '''
        Execute a write statement (INSERT/UPDATE/DELETE)
        
        Options:
        1) Raw SQL + params:
        execute("UPDATE users SET age=? WHERE id=?", (13, 42))

        2) No params:
        execute("DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP")

        Returns:
            int: Number of rows affected (cursor.rowcount).
        '''
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        '''
        Execute a SELECT and return all rows as a list of dicts.

        Options:
        1) Raw SQL + params:
        fetchall("SELECT * FROM users WHERE age >= ?", (13,))

        2) No params:
        fetchall("SELECT id, name FROM users ORDER BY name")

        Returns:
            List[Dict[str, Any]]: Each row converted to a dict (column -> value).
            If no rows match, returns an empty list [].
        '''
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        '''
        Execute a SELECT and return the first row as a dict (or None).

        Options:
        1) Raw SQL + params:
        fetchone("SELECT * FROM users WHERE id = ?", (42,))

        2) No params:
        fetchone("SELECT * FROM users LIMIT 1")

        Returns:
            Optional[Dict[str, Any]]: The first matching row as a dict (column -> value),
            or None if no row matches.
        '''
        with self._cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row is not None else None

    
    # -------------------------
    # Basic CRUD Operations
    # -------------------------

    def create(self, data: SQLInput) -> Dict[str, Any]:
        '''
        Insert rows into table.

        Options:
        1) Raw SQL:
           create(("INSERT INTO t(a,b) VALUES(?,?)", (1,2)))

        2) Dict-based:
           create({
             "table": "users",
             "values": {"name":"Ada", "age": 12}
           })

           Bulk:
           create({
             "table": "users",
             "values": [{"name":"A"}, {"name":"B"}]
           })
        '''
        if isinstance(data, dict):
            table = data["table"]
            values = data["values"]

            if isinstance(values, dict):
                values_list = [values]
            elif isinstance(values, list) and all(isinstance(x, dict) for x in values):
                values_list = values
            else:
                raise ValueError('"values" must be a dict or list[dict].')

            if not values_list:
                return {"rowcount": 0, "lastrowid": None}

            cols = sorted(values_list[0].keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_sql = ", ".join([f'"{c}"' for c in cols])
            sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders});'

            params_list = [tuple(v.get(c) for c in cols) for v in values_list]

            with self._cursor() as cur:
                if len(params_list) == 1:
                    cur.execute(sql, params_list[0])
                    lastrowid = cur.lastrowid
                    return {"rowcount": cur.rowcount, "lastrowid": lastrowid, "sql": sql}
                else:
                    cur.executemany(sql, params_list)
                    return {"rowcount": cur.rowcount, "lastrowid": None, "sql": sql}

        sql, params = self._normalize_sql_input(data, default_params=())
        with self._cursor() as cur:
            cur.execute(sql, params)
            return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid, "sql": sql}

    def update(self, data: SQLInput) -> Dict[str, Any]:
        '''
        Update row(s) in database.

        Options:
        1) Raw SQL:
           update(("UPDATE t SET a=? WHERE id=?", (7, 1)))

        2) Dict-based:
           update({
             "table": "users",
             "set": {"age": 13},
             "where": {"id": 1}
           })
        '''
        if isinstance(data, dict):
            table = data["table"]
            set_vals: Dict[str, Any] = data["set"]
            where_vals: Dict[str, Any] = data.get("where", {})

            if not set_vals:
                raise ValueError('"set" must have at least one column to update.')

            set_cols = sorted(set_vals.keys())
            set_sql = ", ".join([f'"{c}"=?' for c in set_cols])
            params: List[Any] = [set_vals[c] for c in set_cols]

            sql = f'UPDATE "{table}" SET {set_sql}'

            if where_vals:
                where_cols = sorted(where_vals.keys())
                where_sql = " AND ".join([f'"{c}"=?' for c in where_cols])
                sql += f" WHERE {where_sql}"
                params.extend([where_vals[c] for c in where_cols])

            sql += ";"
            rowcount = self.execute(sql, tuple(params))
            return {"rowcount": rowcount, "sql": sql} 
        
        sql, params = self._normalize_sql_input(data, default_params=())
        rowcount = self.execute(sql, params)
        return {"rowcount": rowcount, "sql": sql}

    def delete(self, data: SQLInput) -> Dict[str, Any]:
        '''
        Delete row(s).

        Options:
        1) Raw SQL:
           delete(("DELETE FROM t WHERE id=?", (1,)))

        2) Dict-based:
           delete({
             "table": "users",
             "where": {"id": 1}
           })
        '''
        if isinstance(data, dict):
            table = data["table"]
            where_vals: Dict[str, Any] = data.get("where", {})

            sql = f'DELETE FROM "{table}"'
            params: List[Any] = []

            if where_vals:
                where_cols = sorted(where_vals.keys())
                where_sql = " AND ".join([f'"{c}"=?' for c in where_cols])
                sql += f" WHERE {where_sql}"
                params.extend([where_vals[c] for c in where_cols])
            else:
                # Safety: prevent accidental full-table deletes unless explicitly allowed
                if not data.get("allow_all", False):
                    raise ValueError('Refusing to delete all rows without where. Set "allow_all": True to override.')

            sql += ";"
            rowcount = self.execute(sql, tuple(params))
            return {"rowcount": rowcount, "sql": sql}

        sql, params = self._normalize_sql_input(data, default_params=())
        rowcount = self.execute(sql, params)
        return {"rowcount": rowcount, "sql": sql}

    def query(self, data: SQLInput) -> List[Dict[str, Any]]:
        '''
        Query rows (SELECT).

        Options:
        1) Raw SQL:
           query(("SELECT * FROM users WHERE age>=?", (13,)))

        2) Dict-based:
           query({
             "table": "users",
             "columns": ["id", "name"],
             "where": {"age": 13},
             "order_by": "id DESC",
             "limit": 10,
             "offset": 0
           })
        '''
        if isinstance(data, dict):
            table = data["table"]
            columns = data.get("columns", ["*"])
            where_vals: Dict[str, Any] = data.get("where", {})
            order_by = data.get("order_by")
            limit = data.get("limit")
            offset = data.get("offset")

            col_sql = ", ".join([c if c == "*" else f'"{c}"' for c in columns])
            sql = f'SELECT {col_sql} FROM "{table}"'
            params: List[Any] = []

            if where_vals:
                where_cols = sorted(where_vals.keys())
                where_sql = " AND ".join([f'"{c}"=?' for c in where_cols])
                sql += f" WHERE {where_sql}"
                params.extend([where_vals[c] for c in where_cols])

            if order_by:
                # Order-by is not parameterizable in sqlite. If you allow user input here,
                # validate it against an allowlist.
                sql += f" ORDER BY {order_by}"

            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
                if offset is not None:
                    sql += " OFFSET ?"
                    params.append(offset)

            sql += ";"
            return self.fetchall(sql, tuple(params))

        sql, params = self._normalize_sql_input(data, default_params=())
        return self.fetchall(sql, params)
    

    # -------------------------
    # Helpers
    # -------------------------
    
    def _normalize_sql_input(self, data: SQLInput, default_params: Sequence[Any]) -> Tuple[str, Sequence[Any]]:
        '''
        Ensures one consistent output format.
        '''
        if isinstance(data, str):
            return data, default_params
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
            sql, params = data
            return sql, params
        raise ValueError("SQL input must be a SQL string, (sql, params) tuple, or a dict-based operation.")

