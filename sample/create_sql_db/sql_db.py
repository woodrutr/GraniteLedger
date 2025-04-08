"""DATABASE CODE"""

# import packages
import pandas as pd
from textwrap import dedent
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import db_definitions  # Import your module with table definitions
from db_definitions import Base
import os
from pyomo.common.timing import TicTocTimer
from pathlib import Path


# PREPROCESSOR CODE
def readin_csvs(csv_dir, all_frames):
    for filename in os.listdir(csv_dir):
        # print(filename[:-4])
        f = os.path.join(csv_dir, filename)
        if os.path.isfile(f):
            all_frames[filename[:-4]] = pd.read_csv(f)
            # print(filename[:-4],all_frames[filename[:-4]].columns)

    return all_frames


# DEFINITIONS CODE
def create_pkindx_col(df):
    df = df.reset_index()
    df = df.rename(columns={'index': 'id'})
    df_final = df.set_index('id')
    return df_final


# MAKE CLASS DEFINITIONS
def generate_class_definition(table_name, df):
    class_name = table_name
    index_col = df.index.name
    fields = []

    if index_col is not None:
        dtype = 'Integer'
        if pd.api.types.is_float_dtype(df.index.dtype):
            dtype = 'Float'
        elif pd.api.types.is_string_dtype(df.index.dtype):
            dtype = 'String'
        fields.append(f'{index_col} = Column({dtype}, primary_key=True)')

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            fields.append(f'{col} = Column(DateTime)')
        elif pd.api.types.is_float_dtype(df[col].dtype):
            fields.append(f'{col} = Column(Float)')
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            fields.append(f'{col} = Column(Integer)')
        else:
            fields.append(f'{col} = Column(String)')

    fields_str = '\n    '.join(fields)
    class_definition = f"""
    class {class_name}(Base):
    __tablename__ = '{table_name}'
    {fields_str}
    """
    return class_definition


# WRITE DATA CODE
def create_table_mapping(module):
    """
    Creates a mapping of table names to their SQLAlchemy class definitions
    found in the given module. Assumes that each table class inherits from `Base`
    and that the primary key column is named 'id'.

    Args:
    - module: A module containing SQLAlchemy table class definitions.

    Returns:
    - A dictionary mapping table class names to their class definitions.
    """
    mapping = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Base) and hasattr(obj, '__tablename__'):
            mapping[obj.__tablename__] = obj
    return mapping


# PREPARE DATA
def prepare_dates(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


# LOAD DATA
def load_data_to_db(session, table_class, dataframe):
    dataframe = prepare_dates(dataframe)
    for index, row in dataframe.iterrows():
        data_dict = row.to_dict()
        if dataframe.index.name:
            data_dict[dataframe.index.name] = index

        # Check for existence using the primary key (id), avoiding the costly query operation
        obj = table_class(**data_dict)
        session.merge(
            obj
        )  # `merge` instead of `add` will check and update if exists, otherwise insert

    try:
        session.commit()  # Commit all the operations as a batch
    except IntegrityError:
        session.rollback()


# LOAD DATA
def load_data_to_db_tablewise(session, table_name, table_class, dataframe, engine):
    """Function that checks whether to delete and recreate each table in the cem db file based on a
    series of checks
    - First, check whether names match for existing db and new dataframe; if yes...
    - Check whether table shape is the same; if yes...
    - Check whether data has changed
    At each step, if check fails, delete the data and upload new dataframe. If table scheme (e.g.
    names) don't match, the function deletes and recreates the table schema as well as the data

    Args:
        session (_type_): _description_
        table_name (_type_): _description_
        table_class (_type_): _description_
        dataframe (_type_): _description_
    """

    my_file = Path(dir, 'inputs_database.db')
    if my_file.is_file():
        try:
            dataframe = prepare_dates(dataframe).reset_index()
            table = pd.read_sql_table(table_name, engine.connect())

            ### Check length of columns
            if len(dataframe.columns) == len(table.columns):
                check_names = (dataframe.columns != table.columns).any()
            else:
                check_names = True

        except Exception as e:
            print(f'error with {table_name} loading from db and from dataframes: {str(e)}')
            pass
    else:
        pass

    # TODO: Fix logic behind check_names
    check_names = True

    ### Check names for the tables for schema changes (True means that table names do not match)
    if check_names:
        print(f'New table {table_name} does not match db table; deleting and recreating schema')
        Base.metadata.drop_all(bind=engine, tables=[table_class.__table__])
        Base.metadata.create_all(bind=engine, tables=[table_class.__table__])

        ### Load new data w/ newly created table
        try:
            session.bulk_insert_mappings(table_class, dataframe.to_dict(orient='records'))
            session.commit()
            print(f'Data for {table_name} uploaded to db successfully')
        except IntegrityError:
            session.rollback()
    else:
        print(
            f'Names for columns for {table_name} are the same; checking whether data shape is same'
        )
        if table.shape == dataframe.shape:
            print(f'table shapes for {table_name} are the same; checking whether data matches')
            ### Check if data is the same (True means data doesn't match)
            check_data = (
                ~(dataframe.reset_index(drop=True) == table.reset_index(drop=True)).any().any()
            )
            if check_data:
                print(f'Data does not match for {table_name}; uploading new data')
                try:
                    session.query(table_class).delete()
                    session.commit()
                    session.bulk_insert_mappings(table_class, dataframe.to_dict(orient='records'))
                    session.commit()
                    print(f'Data for {table_name} uploaded to db successfully')
                except IntegrityError:
                    session.rollback()
            else:
                print(f'Data matches for {table_name}; skip without delete and upload')
        else:
            print(f"table shapes for {table_name} aren't the same; delete data and reload")
            try:
                session.query(table_class).delete()
                session.commit()
                session.bulk_insert_mappings(table_class, dataframe.to_dict(orient='records'))
                session.commit()
                print(f'Data for {table_name} uploaded to db successfully')
            except IntegrityError:
                session.rollback()


# EXECUTE FUNCTIONS
def main(dir):
    timer = TicTocTimer()
    timer.tic('start')

    # add csv input files to all frames
    all_frames = {}
    all_frames = readin_csvs(Path(dir, 'inputs'), all_frames)

    for key, df in all_frames.items():
        try:
            all_frames[key] = create_pkindx_col(df)
        except Exception as e:
            print(f'error with {key}: {str(e)}')
            continue

    # declare dicitonary of dataframes
    dataframes = all_frames
    Base = declarative_base()

    # Write definitions to a .py file
    with open(f'{dir}/db_definitions.py', 'w') as f:
        f.write('from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime\n')
        f.write('from sqlalchemy.orm import sessionmaker, declarative_base\n\n')
        f.write('Base = declarative_base()\n\n\n')

        for table_name, df in dataframes.items():
            class_def = generate_class_definition(table_name, df)
            f.write(f'{class_def.strip()}\n\n\n')

        # f.write("\n# Database setup\n")
        # f.write("engine = create_engine('sqlite:///../input/inputs_database.db')\n")
        # f.write("Base.metadata.create_all(engine)\n\n")
        # f.write("# If you need to interact with the database\n")
        # f.write("Session = sessionmaker(bind=engine)\n")
        # f.write("session = Session()\n\n")
        # f.write("# Add data handling below as required\n")
        # f.write("session.close()\n")

    # Initialize database and session
    sql_path = f'sqlite:///{Path(dir, "inputs_database.db")}'
    engine = create_engine(sql_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Use the function to create the table mapping
    table_mapping = create_table_mapping(db_definitions)

    timer.toc('setup')
    # Load data using the mapping
    for table_name, table_class in table_mapping.items():
        dataframe = dataframes[table_name]
        # load_data_to_db(session, table_class, dataframe)
        load_data_to_db_tablewise(session, table_name, table_class, dataframe, engine)
        timer.toc(table_name)

    session.close()
    print('Data has been loaded into the database.')
    timer.toc('end')


if __name__ == '__main__':
    dir = Path(os.path.dirname(os.path.abspath(__file__)))
    main(dir)
