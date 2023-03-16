import psycopg2
import pandas as pd
from psycopg2.extras import NamedTupleCursor
from json import load
# чтение файла конфигурации
def read_config(path):
    cfg = open(path, 'r')
    return load(cfg)

cfg = read_config('config.json')
# подключение к БД
def connect():
    global cfg
    conn = psycopg2.connect(
    host=cfg.get('host'),
    password=cfg.get('password'),
    user=cfg.get('user'),
    port = cfg.get('port'),
    database = cfg.get('database')
    )
    return conn
# создание таблицы
def create_table(conn,name, columns):
    cur = conn.cursor()
    request = f"create table {name} ({columns})"
    cur.execute(request)
    cur.execute("commit")
    cur.close()
# удаление таблицы
def drop_table(conn, name):
    cur = conn.cursor()
    cur.execute(f"drop table {name}")
    cur.execute("commit")
    cur.close()
# получение информации из таблицы
def query(conn, request):
    cur = conn.cursor()
    cur.execute(request)
    out = cur.fetchall()
    cur.close()
    return out
# добавить исторические данные в бд
def insert_ticker_history(conn, ticker, df):
    cur = conn.cursor()
    for _, row in df.iterrows():
        request = f"INSERT INTO {ticker} (TRADEDATE, CLOSE, VOLUME, VALUE, unix_time) VALUES('{row[0]}', {row[1]}, {row[2]}, {row[3]}, {row[4]})"
        cur.execute(request)
        cur.execute('commit')
    cur.close()
# получение исторических данных 
def load_ticker_history(conn, ticker, start_human_readable, end_human_readable):
    start_unix_time = float((pd.to_datetime([start_human_readable]).astype(int) / 10**9)[0])
    end_unix_time = float((pd.to_datetime([end_human_readable]).astype(int) / 10**9)[0])
    out = pd.DataFrame(columns=['TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE', 'unix_time'], data=query(conn, f"select * from {ticker} where unix_time >= {start_unix_time} and unix_time <= {end_unix_time}"))
    return out
# создание таблицы с пользователями
def init_table():
    conn = connect()
    with conn.cursor() as curs:
        curs.execute("create table if not exists users (username VARCHAR(50) PRIMARY KEY, password VARCHAR(36) NOT NULL);")
    conn.commit()
    conn.close()
# вход пользователя
def login(username, password):
    print('login', username, password)
    init_table()
    conn = connect()
    realPass = ''
    try:
        with conn.cursor(cursor_factory=NamedTupleCursor) as curs:
            curs.execute("SELECT password FROM users WHERE username=%s", (username,))
            sgbdrn = curs.fetchone()
            if sgbdrn is None:
                conn.close()
                return False
            realPass = sgbdrn[0]
    except e:
        print(e)
        conn.Close()
        return False
    conn.close()
    
    return password == realPass
# регистрация пользоавтеля
def register(username, password):
    print('register', username, password)
    init_table()
    conn = connect()
    try:
        with     conn.cursor(cursor_factory=NamedTupleCursor) as curs:
            curs.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
    except e:
        print(e)
        conn.close()
        return False
    conn.commit()
    conn.close()
    return True