import socket

def handle_client(client_socket):
    request_data = client_socket.recv(4096)
    if not request_data:
        return

    request_lines = request_data.split(b'\r\n')
    method, url, _ = request_lines[0].split()

    host, port = url.decode().split(b':')
    port = int(port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.connect((host, port))
        server_socket.sendall(request_data)

        response_data = server_socket.recv(4096)

        client_socket.sendall(response_data)

def main():
    proxy_host = 'localhost'
    proxy_port = 8888

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as proxy_socket:
        proxy_socket.bind((proxy_host, proxy_port))
        proxy_socket.listen(5)
        print(f"Proxy server listening on {proxy_host}:{proxy_port}...")

        while True:
            client_socket, client_address = proxy_socket.accept()
            print(f"Connection from {client_address}")

            handle_client(client_socket)

if __name__ == "__main__":
    main()
