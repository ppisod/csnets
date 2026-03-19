using System.Text.Json;

namespace csnets.Visuals.Data;

public class FileSaver {

    public List <float> losses = [];

    public void AddLoss ( float loss ) {
        losses.Add ( loss );
    }

    public void Save ( string filename = "loss-json-" ) {
        if (losses.Count == 0) return;

        string dir = Path.Combine ( Directory.GetCurrentDirectory (), "jsons" );
        Directory.CreateDirectory ( dir );
        string s = DateTime.Now.ToString ( "yyyy-MM-dd-HH-mm-ss" );
        string path = Path.Combine ( dir, filename + s + ".json" );
        File.WriteAllText ( path, JsonSerializer.Serialize ( losses ) );
    }
}