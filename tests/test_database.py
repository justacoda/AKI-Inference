# Relevant imports
import os
import sys
import shutil
import sqlite3
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import SQLite_Manager

class SQLiteManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = SQLite_Manager(db_filepath=self.tmpdir)

    def tearDown(self):
        try:
            self.db.close()
        finally:
            shutil.rmtree(self.tmpdir)

    def test_initialise_database_creates_db_file_in_directory(self):
        self.db._initialise_database()

        expected_path = os.path.join(self.tmpdir, "app.sqlite")
        self.assertEqual(self.db.db_filepath, expected_path)
        self.assertTrue(os.path.exists(expected_path))

        self.assertIsNotNone(self.db._conn)
        self.assertIsInstance(self.db._conn, sqlite3.Connection)


    def test_execute_creates_table_and_inserts(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")

        rc = self.db.execute(
            "INSERT INTO patients (name, age) VALUES (?, ?);",
            ("Ada", 13)
        )
        self.assertEqual(rc, 1)

        rows = self.db.query(("SELECT name, age FROM patients;", ()))
        self.assertEqual(rows, [{"name": "Ada", "age": 13}])
    

    def test_create_dict_inserts_one_row(self):
        self.db.execute(
            "CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);"
        )

        result = self.db.create({"table": "patients", "values": {"name": "Ada", "age": 13}})
        self.assertEqual(result["rowcount"], 1)
        self.assertIsNotNone(result["lastrowid"])

        rows = self.db.query({"table": "patients", "columns": ["name", "age"]})
        self.assertEqual(rows, [{"name": "Ada", "age": 13}])


    def test_create_bulk_inserts_multiple_rows(self):
        self.db.execute(
            "CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);"
        )

        result = self.db.create({
            "table": "patients",
            "values": [
                {"name": "Ben", "age": 14},
                {"name": "Chloe", "age": 13},
                {"name": "Diego", "age": 15},
            ]
        })
        self.assertEqual(result["rowcount"], 3)

        rows = self.db.query(("SELECT name FROM patients ORDER BY id;", ()))
        self.assertEqual([r["name"] for r in rows], ["Ben", "Chloe", "Diego"])


    def test_execute_with_no_params_works(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT);")
        rc = self.db.execute("INSERT INTO patients (id, name) VALUES (1, 'Ada');")
        self.assertEqual(rc, 1)


    def test_fetchall_returns_list_of_dicts(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")
        self.db.execute("INSERT INTO patients (id, name, age) VALUES (?, ?, ?);", (1, "Ada", 13))
        self.db.execute("INSERT INTO patients (id, name, age) VALUES (?, ?, ?);", (2, "Ben", 14))

        rows = self.db.fetchall("SELECT id, name, age FROM patients ORDER BY id;")
        self.assertEqual(rows, [
            {"id": 1, "name": "Ada", "age": 13},
            {"id": 2, "name": "Ben", "age": 14},
        ])
    

    def test_fetchone_with_no_params_works(self):
        row = self.db.fetchone("SELECT 1 AS x;")
        self.assertEqual(row, {"x": 1})


    def test_fetchone_returns_dict_for_first_row(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT);")
        self.db.execute("INSERT INTO patients (id, name) VALUES (?, ?);", (1, "Ada"))
        self.db.execute("INSERT INTO patients (id, name) VALUES (?, ?);", (2, "Ben"))

        row = self.db.fetchone("SELECT id, name FROM patients ORDER BY id;")
        self.assertEqual(row, {"id": 1, "name": "Ada"})
    

    def test_update_dict_changes_matching_rows(self):
        self.db.execute(
            "CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);"
        )
        self.db.create({"table": "patients", "values": {"name": "Ada", "age": 13}})

        result = self.db.update({"table": "patients", "set": {"age": 14}, "where": {"name": "Ada"}})
        self.assertEqual(result["rowcount"], 1)

        row = self.db.fetchone("SELECT age FROM patients WHERE name=?;", ("Ada",))
        self.assertEqual(row, {"age": 14})


    def test_delete_dict_requires_where_by_default(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT);")
        self.db.create({"table": "patients", "values": {"name": "Ada"}})

        with self.assertRaises(ValueError):
            self.db.delete({"table": "patients"})
    

    def test_delete_allow_all_deletes_everything(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);")
        self.db.create({"table": "patients", "values": [{"name": "Ada"}, {"name": "Ben"}]})

        result = self.db.delete({"table": "patients", "allow_all": True})
        # rowcount can vary; assert table is empty
        rows = self.db.query("SELECT * FROM patients;")
        self.assertEqual(rows, [])
    

    def test_query_dict_where_limit_offset_order_by(self):
        self.db.execute(
            "CREATE TABLE patients (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);"
        )
        self.db.create({
            "table": "patients",
            "values": [
                {"name": "A", "age": 13},
                {"name": "B", "age": 13},
                {"name": "C", "age": 13},
                {"name": "D", "age": 14},
            ]
        })

        rows = self.db.query({
            "table": "patients",
            "columns": ["name"],
            "where": {"age": 13},
            "order_by": "id ASC",
            "limit": 2,
            "offset": 1
        })
        self.assertEqual(rows, [{"name": "B"}, {"name": "C"}])
    

    def test_raw_sql_create_and_query_tuple_inputs(self):
        self.db.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT);")

        create_result = self.db.create((
            "INSERT INTO patients (id, name) VALUES (?, ?);",
            (1, "Ada")
        ))
        self.assertEqual(create_result["rowcount"], 1)

        rows = self.db.query(("SELECT id, name FROM patients WHERE id=?;", (1,)))
        self.assertEqual(rows, [{"id": 1, "name": "Ada"}])


    def test_normalize_sql_input_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            self.db._normalize_sql_input(123, default_params=())


    def test_close_resets_connection(self):
        self.db._initialise_database()
        self.assertIsNotNone(self.db._conn)

        self.db.close()
        self.assertIsNone(self.db._conn)

if __name__ == "__main__":
    unittest.main()
