using ScottPlot;
using Range = System.Range;

namespace csnets.Visuals.Plots;

public class LossPlotter {

    private static int[] CreateArray(Range range)
    {
        if (range.Start.IsFromEnd || range.End.IsFromEnd)
            throw new ArgumentException("Cannot use indices from end.", nameof(range));

        var start = range.Start.Value;
        var end = range.End.Value;

        if (end < start)
            throw new ArgumentException("The end cannot be smaller than the start.", nameof(range));

        var length = end - start + 1;
        var array = new int[length];

        for (int idx = 0, value = start; idx < length; idx++, value++)
            array[idx] = value;

        return array;
    }

    public List <float> losses = [];

    public void AddLoss ( float loss ) {
        losses.Add ( loss );
    }

    public SavedImageInfo? Save ( string filename = "loss" ) {
        try
        {
            if (losses.Count == 0) return null;

            Plot plot = new Plot ();
            plot.Add.Scatter ( CreateArray ( new Range ( 0, losses.Count - 1 ) ), losses.ToArray () );

            string dir = Path.Combine ( Directory.GetCurrentDirectory (), "graphs" );
            Directory.CreateDirectory ( dir );
            DateTime now = DateTime.Now;
            string s = now.ToString ( "yyyy-MM-dd HH-mm-ss" );
            SavedImageInfo info = plot.SavePng ( Path.Combine ( dir, filename + s + ".png" ), 1400, 1000 );

            return info;

        } catch (Exception e) {
            Console.WriteLine ( e );
            return null;
        }

    }
}