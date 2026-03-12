using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public interface INet {
    public FeedForwardNet net { get; set; }
    abstract void Train ( int epochs );
    abstract float[] Run ( float[] inputs );
}