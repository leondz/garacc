#!/usr/bin/env python3

"""garak listener

Service that listens on test ports, stores activity, and returns summary results.
The listener has a service port that processes instructions and relays results.
Only one set of results is stored at a time.

Results are intentionally ephemeral.

Invocation from garak source root:

  python tools/glisten.py

Service protocol instructions:
 
 INFO

  * return as JSON ``{"status": {"code":0, "message":"OK"}, "version":"glisten v.xxxxx"}``

 START <id str> <port spec>

  * discard all current results
  * try to bind to ports in port spec (comma separated list) (max #MAX_PORTS_LISTENED ports)
  * listen for connection attempts and content on these ports
  * return dict as JSON,  ``{"status": {"code":1, "message":"garak listen started"}``

 COLLECT <id str>
  
  * if <id str> does not match the current run or there is no current run, return
    in JSON ``{"status": {"code": 2, "message": "no run under that ID"}}``
  * if <id str> does match the current run,
    
    * return a dict with:

      * "status" of code 3, message "ending run" 
      * "results" which is a list, each entry being:

        * "port" with port number
        * "bound" with True of False, relaying whether binding worked
        * if bound is True:

            * "opened" which is True or False
            * "content" which is a list of first ``MAX_CONTENT_LOGGED`` bytes of content sent

  * discard all current results ``{"status": {"code":3, "message":"unrecognised command"}``

 Other and invalid inputs return {} 

Note that START can always be used to "flush" the service, even with the same ID.

Recommend a slight delay/backoff after a first failed COLLECT attempt, to mitigate ID brute forcing.

A local log can collect info on whether the server could bind ports, and on service instructions received
"""

import json
import logging
import selectors
import socket
import time
import types

SERVICE_PORT = 9218
HOST = "127.0.0.1"
MAX_CONTENT_LOGGED = 131072  # bytes
MAX_PORTS_LISTENED = 250
MAX_LISTENERS_PER_PORT = 32  # this might like to be in the region of parallel_attempts


class GarakListener:

    def __init__(self, config=dict()) -> types.NoneType:
        self._reset_session()

    def _accept_wrapper(self, sock):
        conn, addr = sock.accept()
        local_addr, local_port = conn.getsockname()
        logging.info(f"accepted conxn from {addr} on port {local_port}")
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, events, data=data)

    def _send_as_json(self, obj, sock):
        obj_str = json.dumps(obj)
        sent = sock.send(bytes(obj_str + "\n", encoding="utf-8"))
        return sent

    def _serve_service_connection(self, key, mask):
        data = key.data
        sock = key.fileobj
        instruction = ""
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)
            if recv_data:
                instruction = recv_data
            else:
                logging.info(f"closing conxn to {data.addr}")
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            instruction_parts = instruction.strip()
            if not instruction_parts:
                return
            instruction_parts = instruction_parts.decode("utf-8").split()
            print(instruction_parts)
            if instruction_parts[0] == "INFO":
                logging.info(f"reporting to {data.addr}")
                msg_obj = {
                    "status": {"code": 0, "message": "OK"},
                    "version": f"glisten v.0.0.0",
                }
                sent = self._send_as_json(msg_obj, sock)
            elif instruction_parts[0] == "START":
                if len(instruction_parts) != 3:
                    self._send_err(sock, explanation="START takes two args")
                else:
                    self.session_id = instruction_parts[1]
                    self.delay = False
                    msg_obj = {"status": {"code": 1, "message": "garak listen started"}}
                    self.status["session_id"] = self.session_id
                    self._send_as_json(msg_obj, sock)

            elif instruction_parts[0] == "COLLECT":
                if len(instruction_parts) != 2:
                    self._send_err(sock, "COLLECT takes one arg")
                else:
                    if instruction_parts[1] == self.session_id:
                        print(self.status)
                        self._send_as_json(self.status, sock)
                        self._reset_session()
                    else:
                        msg_obj = {
                            "status": {"code": 2, "message": "no run under that ID"}
                        }
                        self._send_as_json(msg_obj, sock)
                        if self.delay:
                            time.sleep(
                                0.15
                            )  # praying this blocks the thread and not service

            else:
                self._send_err(sock)

    def _serve_connection(self, key, mask):
        local_host, local_port = key.fileobj.getsockname()
        if local_port == SERVICE_PORT:
            self._serve_service_connection(key, mask)

    def _reset_session(self):
        self.delay = False
        self.session_id = None
        self.status = {}
        self.test_sockets = []

    def _send_err(self, sock, explanation: str = ""):
        msg_obj = {"status": {"code": 3, "message": "unrecognised command"}}
        if explanation:
            msg_obj["status"]["message"] += f": {explanation}"
        sent = self._send_as_json(msg_obj, sock)
        return sent

    def _init_service(self, port=SERVICE_PORT):

        self.sel = selectors.DefaultSelector()

        s_service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_service.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s_service.bind((HOST, port))
        logging.info(f"service bound to {HOST}:{port}")
        s_service.listen()
        logging.info(f"service listening")
        s_service.setblocking(False)
        self.sel.register(s_service, selectors.EVENT_READ, data=None)

        try:
            while True:
                events = self.sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        self._accept_wrapper(key.fileobj)
                    else:
                        self._serve_connection(key, mask)
        except KeyboardInterrupt:
            logging.info("Caught keyboard interrupt, exiting")
        finally:
            self.sel.close()

    def service(
        self,
    ):
        logging.info("starting")
        self._init_service()


if __name__ == "__main__":
    glisten = GarakListener()
    glisten.service()
