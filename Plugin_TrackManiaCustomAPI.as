#name "Trackmania Custom API"
#author "Jack Venberg"

#category "Utility"

void Main()
{
    // Create a new socket.
    auto sock = Net::Socket();

    // Try to initiate a socket to ip.mrag.nl on port 80.
    if (!sock.Connect("127.0.0.1", 65432)) {
        // If it failed, there was some socket error. (This is not necessarily
        // a connection error!)
        print("Couldn't initiate socket connection.");
        return;
    }

    print("Connecting to host...");

    // Wait until we are connected. This is indicated by whether we can write
    // to the socket.
    while (!sock.CanWrite()) {
        yield();
    }

    print("Connected! Sending request...");

    // Send raw data (as a string) to the server.
    if (!sock.WriteRaw("Test\r\n")) {
        // If this fails, the socket might not be open. Something is wrong!
        print("Couldn't send data.");
        return;
    }

    // We're all done!
    print("All done!");
    // print("Response: \"" + response + "\"");

    // Close the socket.
    sock.Close();
}