import airsim
import time 
import csv
import math 
import os
from datetime import datetime

client = airsim.MultirotorClient()
client.confirmConnection()

object_names=client.simListsceneObjects