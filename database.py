import sqlite3
import os
  
# Function for Convert Binary Data 
# to Human Readable Format
def convertToBinaryData(filename):
      
    # Convert binary format to images 
    # or files data
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData
 
dbpath =  os.path.join(os.path.dirname(__file__),'assets.db')
sqliteConnection = sqlite3.connect(dbpath)

  
def insertBLOB(name, photo):
    res=False
    global sqliteConnection
    try:
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
          
        # insert query
        sqlite_insert_blob_query = """ INSERT INTO images
                                  (filename, image) VALUES (?, ?)"""
          
        # Converting human readable file into 
        # binary data
        Photo = convertToBinaryData(photo)
          
        # Convert data into tuple format
        data_tuple = (name, Photo)
          
        # using cursor object executing our query
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()
        res = True
  
    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
      
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
    return res    

 
def Get_Image(name):
    res=(0,0)
    global sqliteConnection
    try:
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
          
        # insert query
        sqlite_extract_blob_query = """SELECT * FROM images WHERE filename=(?)"""
          

        cursor.execute(sqlite_extract_blob_query, (name,))
        res = cursor.fetchone()
        sqliteConnection.commit()
        print("Fetched Image")
        cursor.close()
  
    except sqlite3.Error as error:
        print("Failed to fetch blob data into sqlite table", error)
      
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
    return res
    

def get_images_name():
    names = []
    global sqliteConnection
    try:
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")
          
        # insert query
        sqlite_extract_blob_query = """SELECT * FROM images"""

        cursor.execute(sqlite_extract_blob_query)
        res = cursor.fetchall()
        for img in res: names.append(img[0])

        sqliteConnection.commit()
        print("GOT Images filename all *")
        cursor.close()
  
    except sqlite3.Error as error:
        print("Failed to fetch filenames data into sqlite table", error)
      
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")
    return names
    
