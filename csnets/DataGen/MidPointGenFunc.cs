namespace csnets.DataGen;

public struct MidPointGenFuncOutput {
    public float in_A;
    public float in_B;
    public float out_M;
}

public class MidPointGenFunc {

    public float c = 0;
    public float m = 0;

    public float range = 5;
    public float offset = 3;

    public Random random = new ();

    public float getV ( float x ) {
        return m * x + c;
    }

    public MidPointGenFuncOutput gen () {
        MidPointGenFuncOutput output = new ()
        {
            in_A = (float) ( random.NextDouble () * range ) + offset,
            in_B = (float) ( random.NextDouble () * range ) + offset
        };
        output.out_M = getV ( output.in_A + output.in_B / 2 );
        return output;
    }

    public List <MidPointGenFuncOutput> gen ( int k ) {
        List <MidPointGenFuncOutput> q = [];
        for (int i = 0; i < k; i++)
        {
            q.Add (gen ());
        }

        return q;
    }

}