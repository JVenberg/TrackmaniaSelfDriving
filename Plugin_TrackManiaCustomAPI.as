#name "Trackmania Custom API"
#author "Jack Venberg"

#category "Utility"


Net::Socket@ sock;

void Main()
{   
    
    int prevRaceTime = -1;
    ConnectSocket();
    while (true) {
        CTrackMania@ app = cast<CTrackMania>(GetApp());
        CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
        if (playground !is null) {
            if (playground.GameTerminals.Length > 0) {
                CGameTerminal@ terminal = cast<CGameTerminal>(playground.GameTerminals[0]);
                CSmPlayer@ player = cast<CSmPlayer>(terminal.GUIPlayer);
                if (player !is null) {
                    CSmScriptPlayer@ scriptApi = cast<CSmScriptPlayer>(player.ScriptAPI);
                    if (scriptApi !is null) {
                        int raceTime = scriptApi.CurrentRaceTime;
                        if (raceTime > prevRaceTime && raceTime > 0) {
                            float speed = scriptApi.Speed * 3.6f;
                            print('' + speed);
                            if (!sock.Write(speed)) {
                                ConnectSocket();
                                sleep(50);
                                continue;
                            }
                        }
                        prevRaceTime = raceTime;
                    } 
                }
                
            }
        }
        
        yield();
    }

    // We're all done!
    print("All done!");
    sock.Close();
}

void ConnectSocket() {
    if (sock !is null) {
        sock.Close();
    }
    @sock = Net::Socket();
    sock.Connect("127.0.0.1", 65432);
}
