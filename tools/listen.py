#!/usr/bin/env python3

"""garak port listener

Service that listens on ports, stores activity, and returns summary results.
The listener has a service port that processes instructions and relays results.
Only one set of results is stored at a time.

Results are intentionally ephemeral.

service instructions:
 
 INFO

  * return as JSON ``{"status": {"code":0, message:"OK"}, "version":"garak v.xxxxx"}``

 START <id str> <port spec>

  * discard all current results
  * try to bind to ports in port spec (comma separated list) (max #MAX_PORTS_LISTENED ports)
  * listen for connection attempts and content on these ports
  * return dict as JSON,  ``{"status": {"code":1, "message":"garak listen started"}``

 COLLECT <id str>
  
  * if <id str> does not match the current run or there is no current run, return
    in JSON ``{"status": {"code": 2, "message": "no run under that ID"}}
  * if <id str> does match the current run,
    
    * return a dict with:

      * "status" of code 3, message "ending run" 
      * "results" which is a list, each entry being:

        * "port" with port number
        * "bound" with True of False, relaying whether binding worked
        * if bound is True:

            * "opened" which is True or False
            * "content" which is a list of first ``MAX_CONTENT_LOGGED`` bytes of content sent

  * discard all current results

Note that START can always be used to "flush" the service, even with the same ID.

Recommend a slight delay/backoff after a first failed COLLECT attempt, to mitigate ID brute forcing.

A local log can collect info on whether the server could bind ports, and on service instructions received
"""

import logging
import selectors
import socket
import types

SERVICE_PORT = 9218
HOST = "127.0.0.1"
MAX_CONTENT_LOGGED = 131072  # bytes
MAX_PORTS_LISTENED = 250
MAX_LISTENERS_PER_PORT = 32  # this might like to be in the region of parallel_attempts


def _start_service(port=SERVICE_PORT):

    def service_accept_wrapper(sock):
        conn, addr = sock.accept()
        logging.info(f"accepted conxn from {addr}")
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        sel.register(conn, events, data=data)

    def service_serve_connection(key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(4096)
            if recv_data:
                data.outb += recv_data
            else:
                logging.info(f"closing conxn to {data.addr}")
                sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if data.outb:
                logging.info(f"echoing to {data.addr}")
                sent = sock.send(data.outb)
                data.outb = data.outb[sent:]

    sel = selectors.DefaultSelector()

    s_service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_service.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s_service.bind((HOST, port))
    logging.info(f"service bound to {HOST}:{port}")
    s_service.listen()
    logging.info(f"service listening")
    s_service.setblocking(False)
    sel.register(s_service, selectors.EVENT_READ, data=None)

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    service_accept_wrapper(key.fileobj)
                else:
                    service_serve_connection(key, mask)
    except KeyboardInterrupt:
        logging.info("Caught keyboard interrupt, exiting")
    finally:
        sel.close()


def service():
    logging.info("starting")
    _start_service()


if __name__ == "__main__":
    service()
