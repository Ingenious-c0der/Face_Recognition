import cv2
import face_recognition
import numpy as np
import motor
from motor.motor_asyncio import AsyncIOMotorClient,AsyncIOMotorDatabase
import ssl 
from bson.binary import Binary
import pickle
Mongo_Client =motor.motor_asyncio.AsyncIOMotorClient(r"mongodb+srv://<username>:<password>@<dbname>.si3ce.mongodb.net/Bot_data?retryWrites=true&w=majority",ssl_cert_reqs=ssl.CERT_NONE, serverSelectionTimeoutMS=5000)
db = Mongo_Client.get_database('<dbname>')

async def img_ready(imagepath):
    frame = cv2.imread(imagepath)
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    return small_frame[:, :, ::-1]



async def match_encoding(imagepath)->(Bool,None):
    """
    Function which actually matches the face of the person in the image with the face encodings in the database
    Parameters : imagepath (this will change according to how js file communicates with py file),knownface encodings from the db
    Returns : True If there is a match in the database for the given image
              False if there is no match in the db , also saves the generated encoding to database
    """
    known_face_encodings = await Mongo_functions.get_all_encodings()
    
    image = await img_ready(imagepath)
    face_locations = face_recognition.face_locations(image)
    face_encoding_to_check =  np.asarray(face_recognition.face_encodings(image, face_locations))

    face_encoding_to_check = np.asarray(face_recognition.face_encodings(image, face_locations)[0])
    
    
    comparison_list = face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)
    return True if comparison_list.count(True)>=1 else False,await Mongo_functions.push_encoding_to_db(face_encoding_to_check)

class Mongo_functions:
    current_db = db.Encodings
    async def push_encoding_to_db(encoding)->None:
        """Function that saves/writes the generated encoding to the database in the form of Binaries
        Parameters : encoding (to be stored)
        Returns :None


        The current storage format is 
        _id:        random object generated id
        encoding : the face encoding of that person 

        ...
        Further expected modification -> Adding a age filter: age of that person 
        It makes sense to store the images in certain age group (each age group having its own collection in the db) which would 
        make the searching process lighter and faster
         """
        await Mongo_functions.current_db.insert_one({"encoding":Binary(pickle.dumps(encoding, protocol=2), subtype=128 )})

    async def get_all_encodings()->list[np.array]:
        """
        Function which returns all the encodings from the database in the format of list of np arrays

        Parameters : None  , in future  (expected parameter) : age (to filter accoirding to age group)
        
        """
        all_encodings = Mongo_functions.current_db.find({},{"encoding":1,"_id":0})
        decoded_encodings = []
        async for encoding_dict in all_encodings:
            decoded_encodings.append(pickle.loads(encoding_dict['encoding']))
        return decoded_encodings



