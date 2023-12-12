# server.py
import flwr as fl

def main():
    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10)
    )

if __name__ == "__main__":
    main()
