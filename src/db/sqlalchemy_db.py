from utils import common_utils
from sqlalchemy import create_engine
from sqlalchemy import text


class SqliteAlchemyDatabase:
    def __init__(self, db_name):
        self.logger = common_utils.get_loguru()

        self.pool_size = 10
        self.max_overflow = 20

        self.sqlalchemy_url = "sqlite:///{}".format(db_name)
        self.engine = create_engine(
            self.sqlalchemy_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
        )


class DialogueAlchemyDatabase(SqliteAlchemyDatabase):
    def __init__(self, db_name):
        super().__init__(db_name)

        self._init_dialogue_data()

    def _init_dialogue_data(self):
        with self.engine.connect() as connect:
            sql_text = text(
                """CREATE TABLE IF NOT EXISTS dialogue(
                    id TEXT PRIMARY KEY NOT NULL,
                    data TEXT NOT NULL,
                    index_field TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    state INT NOT NULL);"""
            )

            connect.execute(sql_text)
            connect.commit()
            self.logger.info("Init Dialogue Table")

    def get_all_dialogue(self):
        results = []
        with self.engine.connect() as connect:
            sql_text = text(
                """SELECT id, data, index_field, owner, state FROM dialogue"""
            )
            self.logger.debug("SQL EXEC {}".format(sql_text))

            rows = connect.execute(sql_text)
            for row in rows:
                result = {
                    "id": row[0],
                    "data": row[1],
                    "index_field": row[2],
                    "owner": row[3],
                    "state": row[4],
                }
                results.append(result)
        return results

    def get_dialogue_size(self):
        with self.engine.connect() as connect:
            sql_text = text("""SELECT COUNT(id) FROM dialogue""")

            self.logger.debug("SQL EXEC {}".format(sql_text))

            rows = connect.execute(sql_text)
            for row in rows:
                return row[0]
        return 0

    def get_dialogue_by_id(self, session_id):
        with self.engine.connect() as connect:
            sql_text = text(
                """SELECT data, index_field, owner, state FROM dialogue WHERE id=:id"""
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {}".format(session_id))

            param = {"id": id}
            row = connect.execute(sql_text, param)
            result = {
                "data": row[0][0],
                "index_field": row[0][1],
                "owner": row[0][2],
                "state": row[0][3],
            }
            return result

    def save_dialogue(self, session_id, data, index_field, owner, state):
        with self.engine.connect() as connect:
            sql_text = text(
                """INSERT INTO dialogue (id, data, index_field, owner, state) 
                                VALUES (:id, :data, :index_field, :owner, :state)"""
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug(
                "SQL PARAM {},{},{},{},{}".format(
                    session_id, data, index_field, owner, state
                )
            )

            param = {
                "id": session_id,
                "data": data,
                "index_field": index_field,
                "owner": owner,
                "state": state,
            }
            connect.execute(sql_text, param)
            connect.commit()

    def update_dialogue(self, session_id, data):
        with self.engine.connect() as connect:
            sql_text = text("""UPDATE dialogue SET data=:data WHERE id=:id""")

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {}".format(session_id))

            param = {"id": session_id, "data": data}
            connect.execute(sql_text, param)
            connect.commit()


class TaskAlchemyDatabase(SqliteAlchemyDatabase):
    def __init__(self, db_name):
        super().__init__(db_name)

        self.logger.info("Init EvalAlchemyDatabase {}.".format(db_name))

    def init_table(self, table_name, feild_list):
        with self.engine.connect() as connect:
            raw_text = """CREATE TABLE IF NOT EXISTS {}(
                    session_id TEXT PRIMARY KEY NOT NULL""".format(
                table_name
            )

            for feild in feild_list:
                raw_text += """,{}  TEXT""".format(feild)
            raw_text += """);"""

            self.logger.info("Init {} Table {}".format(table_name, raw_text))

            sql_text = text(raw_text)
            connect.execute(sql_text)
            connect.commit()

    def get_feild_by_id(self, table_name, session_id, feild_name):
        with self.engine.connect() as connect:
            sql_text = text(
                """SELECT {} FROM {} WHERE session_id= :session_id""".format(
                    feild_name, table_name
                )
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {}".format(session_id))

            param = {"session_id": session_id}
            rows = connect.execute(sql_text, param)
            result = {}
            for row in rows:
                if row[0]:
                    result = {feild_name: row[0]}
            return result

    def exist(self, table_name, session_id):
        with self.engine.connect() as connect:
            sql_text = text(
                """SELECT COUNT(*) FROM {} WHERE session_id= :session_id""".format(
                    table_name
                )
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {}".format(session_id))

            param = {"session_id": session_id}
            rows = connect.execute(sql_text, param)
            if rows.fetchone()[0]:
                return True
            else:
                return False

    def save_feild(self, table_name, session_id, feild_name, feild_info):
        with self.engine.connect() as connect:
            sql_text = text(
                """INSERT INTO {} (session_id, {}) VALUES (:session_id, :{})""".format(
                    table_name, feild_name, feild_name
                )
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {},{}".format(session_id, feild_info))

            param = {"session_id": session_id, feild_name: feild_info}
            connect.execute(sql_text, param)
            connect.commit()

    def update_feild(self, table_name, session_id, feild_name, feild_info):
        with self.engine.connect() as connect:
            sql_text = text(
                """UPDATE {} SET {}=:{} WHERE session_id=:session_id""".format(
                    table_name, feild_name, feild_name
                )
            )

            self.logger.debug("SQL EXEC {}".format(sql_text))
            self.logger.debug("SQL PARAM {},{}".format(feild_info, session_id))

            param = {"session_id": session_id, feild_name: feild_info}
            connect.execute(sql_text, param)
            connect.commit()
