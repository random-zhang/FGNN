import sqlite3

def createSelectedTable(table_name='selected'):
    conn = sqlite3.connect('system.db')
    cursor = conn.cursor()
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        task_name TEXT ,
        label_name TEXT,
        s_no INTEGER,
        selected TEXT 
    )
    '''
    cursor.execute(create_table_query)
    conn.commit()
    conn.close()
def insert_fp(task_name,selected,label_name='Class',table_name='fp'):
    # 连接到数据库
    conn = sqlite3.connect('system.db')

    # 创建一个光标对象
    cursor = conn.cursor()

    # 编写SQL语句来插入数据
    insert_query = f"INSERT INTO {table_name} (task_name,label_name, selected) VALUES (?,?, ?)"
    data = (task_name, label_name,selected)

    # 执行SQL语句插入数据
    cursor.execute(insert_query, data)
    # 提交更改并关闭连接
    conn.commit()
    conn.close()



def read_from_selectecd(sort,task_name,s_no,label_name='Class',table_name='selected'):
    conn = sqlite3.connect('system.db')
    # 创建一个光标对象
    cursor = conn.cursor()
    select_query = f"SELECT * FROM {table_name} where task_name =? and s_no =? and label_name= ?"

    # 执行查询语句
    cursor.execute(select_query, (task_name,s_no,label_name))
    result=None
    # 获取查询结果
    results = cursor.fetchone()
    if results is not None:
       result=results[-1]
    conn.close()
    if sort:
        return list(sorted(eval(result)))
    return eval(result)
def updata_selected(task_name,s_no,selected,label_name='Class',table_name='selected'):
    conn = sqlite3.connect('system.db')
    # 创建一个光标对象
    cursor = conn.cursor()
    cursor.execute(f"UPDATE {table_name} SET selected = ? WHERE task_name = ? and s_no=? and label_name =?", (selected, task_name,s_no,label_name))
    conn.commit()
    conn.close()
def insert_selected(task_name,s_no,selected,label_name='Class',table_name='selected'):
    # 连接到数据库
    conn = sqlite3.connect('system.db')

    # 创建一个光标对象
    cursor = conn.cursor()

    # 编写SQL语句来插入数据
    insert_query = f"INSERT INTO {table_name} (task_name, s_no,selected,label_name) VALUES (?, ?,?,?)"
    data = (task_name, s_no,selected,label_name)

    # 执行SQL语句插入数据
    cursor.execute(insert_query, data)
    # 提交更改并关闭连接
    conn.commit()
    conn.close()
# createSelectedTable()
# insert_selected('bbbp',100,str([1,2,3,5,6,7]))
#
labels=[ 'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'General disorders and administration site conditions',
                 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                 'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations',
                 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders',
                 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders',
                 'Nervous system disorders', 'Injury, poisoning and procedural complications']
# for label in labels:
#     try:
#
#       a = read_from_selectecd(False, 'sider', 2400, label_name=label, table_name='selected')
#       print(a)
#     except TypeError as e:
#         print(label)
    # print(a)
# #
# a=read_from_selectecd(False,'hiv',2400,label_name='Class',table_name='mim')
# print(a)
# updata_selected('bbbp',100,str([1,2,6,7,9,3,5,6,7]))
# print(read_from_selectecd('bbbp',100))
# print(read_from_selectecd('esol',1000))