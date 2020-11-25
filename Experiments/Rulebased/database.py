# layout will be
# | num_players | agent | turn | state | action | team |
# for the state action table and
# | num_players | turn | state | team |
# for the state table
# todo maybe we leave the state table out for now,
#  because we can deduce it from the complete state action table
import sqlite3

def create_connection(path):
    """ create a database connection to the SQLite database
            specified by path
        :param path: database file
        :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(str(path))
        return conn
    except sqlite3.Error as e:
        print(e)
    return None

def insert_dictionary_data(conn, replay_dictionary):
    """
    CREATE TABLE IF NOT EXISTS info (PRIMARY KEY id int, username text, password text)
    replay_dictionary example:
    dict_keys(['OuterAgent', 'FlawedAgent', 'IGGIAgent', 'team', 'num_players'])
    replay_dictionary['OuterAgent'] : dict_keys(['states', 'actions', 'turns'])
    - replay_dictionary['OuterAgent']['states']: List[List[int]]
    - replay_dictionary['OuterAgent']['actions']: List[int]
    - replay_dictionary['OuterAgent']['turns']: List[int]
    """
    cursor = conn.cursor()
    # create table if it does not exist already
    # | num_players | agent | turn | state | action | team |
    cursor.execute(''' CREATE TABLE IF NOT EXISTS pool_of_states( 
    num_players INTEGER , agent TEXT, turn INTEGER, state TEXT, action INTEGER, team TEXT)''')
    # conn.commit()

    num_players = replay_dictionary.pop('num_players')
    team = str(replay_dictionary.pop('team'))
    values = []
    for agent in replay_dictionary.keys():
        num_transitions = len(replay_dictionary[agent]['turns'])
        for i in range(num_transitions):
            # | num_players | agent | turn | state | action | team |
            row = (num_players,
                   agent,
                   replay_dictionary[agent]['turns'][i],
                   str(replay_dictionary[agent]['states'][i]),
                   replay_dictionary[agent]['actions'][i],
                   team
                   )
            assert type(row[4]) == int, f'action was {row[4]}'
            assert 0 <= row[4] <= 50, f'action was {row[4]}'
            values.append(row)
    cursor.executemany('INSERT INTO pool_of_states VALUES (?,?,?,?,?,?)', values)
    conn.commit()
