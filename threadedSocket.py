# import socket  # Import socket module
import websockets  # Import socket module
import threading  # Handle Threads
import time
import json

lock = threading.Lock()
class ThreadedSocket():
    def __init__(self):
        self.data = None
        self.frame = None
        try:
            self.s = websockets.serve(self.sendData, '0.0.0.0', 5679)   # Create a socket object
        except Exception as e:
            print(e)

    def handle(self, connAddr):
        print("New connection received")
        conn = connAddr[0]
        while True:
            conn.send("ok".encode())
            # with lock:
                # if self.data is not None:
                    # conn.send(json.dumps(self.data).encode())

                # if self.frame is not None:
                #     conn.send(self.frame)

    def start(self):
        print ('Server started!')
        print ('Waiting for clients...')
        while True:
            t = threading.Thread(target=self.handle, args=(self.s.accept(), ))
            t.start()

    def updateData(self, data):
        self.data = data

    def updateFrame(self, frame):
        self.frame = frame