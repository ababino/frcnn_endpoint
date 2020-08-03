from flask_restplus import Api
from .frcnn import api as ns1

api = Api(
    title='My Title',
    version='1.0',
    description='A description',
    # All API metadatas
)

api.add_namespace(ns1)
