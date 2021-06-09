#name "Trackmania Custom API"
#author "Jack Venberg"

#category "Utility"

int PORT = 65432;
string URL = "127.0.0.1";

Net::Socket@ sock;

void Main()
{

    bool paused = true;
    ConnectSocket();
    while (true) {
        CSmScriptPlayer@ scriptApi = GetPlayerScriptAPI();
        CGameManiaPlanetScriptAPI@ maniaApi = GetManiaScriptAPI();
        if (scriptApi !is null) {
            int raceTime = scriptApi.CurrentRaceTime;
            auto json = Json::Object();
            if (raceTime > 0 && !maniaApi.ActiveContext_InGameMenuDisplayed) {
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

        yield();
    }

    print("All done!");
    sock.Close();
}

CGameManiaPlanetScriptAPI@ GetManiaScriptAPI() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    return app.ManiaPlanetScriptAPI;
}

CSmScriptPlayer@ GetPlayerScriptAPI() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground !is null) {
        if (playground.GameTerminals.Length > 0) {
            CGameTerminal@ terminal = cast<CGameTerminal>(playground.GameTerminals[0]);
            CSmPlayer@ player = cast<CSmPlayer>(terminal.GUIPlayer);
            if (player !is null) {
                return cast<CSmScriptPlayer>(player.ScriptAPI);
            }
        }
    }
    return null;
}

void ConnectSocket() {
    if (sock !is null) {
        sock.Close();
    }
    @sock = Net::Socket();
    sock.Connect(URL, PORT);
}
