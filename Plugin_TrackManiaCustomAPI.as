#name "Trackmania Custom API"
#author "Jack Venberg"

#category "Utility"


Net::Socket@ sock;

void Main()
{   
    
    int prevRaceTime = -1;
    bool paused = true;
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
                        auto json = Json::Object();
                        if (raceTime < 0) {
                            prevRaceTime = raceTime;
                        }
                        if (raceTime > prevRaceTime && raceTime > 0) {
                            if (paused) {
                                sleep(500);
                            }
                            json['speed'] = Json::Value(scriptApi.Speed * 3.6f);
                            if (!sock.WriteRaw(Json::Write(json) + '\n')) {
                                ConnectSocket();
                                yield();
                                continue;
                            }
                            print(Json::Write(json));
                            prevRaceTime = raceTime;
                            paused = false;
                        } else if (!paused) {
                            paused = true;
                            json['speed'] = Json::Value();
                            if (!sock.WriteRaw(Json::Write(json) + '\n')) {
                                ConnectSocket();
                                yield();
                                continue;
                            }
                            print(Json::Write(json));
                        }
                        
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
