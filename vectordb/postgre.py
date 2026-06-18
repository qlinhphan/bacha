import psycopg2
import psycopg2.extras

def init_postgre():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="WikiDB",
        user="admin",
        password="admin123"
    )
    cursor = conn.cursor()

    return cursor

def create_table_postgre(cursor):
    create_table_query = """
        CREATE TABLE IF NOT EXISTS history (
            id SERIAL PRIMARY KEY,
            sessionId VARCHAR(100) NOT NULL,
            userId VARCHAR(100) NOT NULL,
            content TEXT NOT NULL
        );
    """
    cursor.execute(create_table_query)
    cursor.connection.commit()

def save_data_into_postgre(cursor, sessionId, userId, content):
    insert_query = """
        INSERT INTO history (sessionId, userId, content) 
        VALUES (%s, %s, %s);
    """
    data_to_insert = (sessionId, userId, content) 
    
    cursor.execute(insert_query, data_to_insert)
    cursor.connection.commit()