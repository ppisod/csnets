using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public class MNISTNetwork : INet {

    public FeedForwardNet net { get; set; }
    public void Train ( int epochs ) {
        throw new NotImplementedException ();
    }
    public float[] Run ( float[] inputs ) {
        throw new NotImplementedException ();
    }
    public float Run ( ) {
        throw new NotImplementedException ();
    }

}