'''A Flask app to serve the model results'''
import os
from flask import Flask, request
from flask_restful import Api, Resource
from dotenv import load_dotenv
from doctors_ai.model import output
from doctors_ai.tools.definitions import MINIO_API_HOST, MINIO_SECRET_KEY, MINIO_ACCESS_KEY
load_dotenv()

app = Flask(__name__)
api = Api(app)

DB_LOCATION = os.environ.get("DB_LOCATION")
MODEL_LOCATION = os.environ.get("MODEL_LOCATION")


class Preds(Resource): #type: ignore
    """
    A class to serve predctions

    Methods
    -------
    post()
        Gets the predicted number of admission and number in ed by date
    """

    def post(self) -> tuple[dict[str, int], int]:
        """
        Returns the JSON for predicted number of admissions and number in ed by date
        """
        json_ = request.json
        date = json_['date']
        # Make predictions using date
        predicted_number_admissions, number_in_ed = \
            output(date, MINIO_API_HOST, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, DB_LOCATION)
        res = {'predicted_number_admissions': predicted_number_admissions,
               'number_in_ed': number_in_ed}
        return res, 200  # Send the response object

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
    app.run(debug=True, port=3000)
