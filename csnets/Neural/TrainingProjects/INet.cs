using csnets.Neural.Networks;

namespace csnets.Neural.TrainingProjects;

public interface INet {
    /// <summary>
    /// the network
    /// </summary>
    public FeedForwardNet net { get; set; }

    /// <summary>
    /// trains the network
    /// </summary>
    /// <param name="epochs">how many epochs</param>
    abstract void Train ( int epochs );

    /// <summary>
    /// run on custom input
    /// </summary>
    /// <param name="inputs"> float inputs </param>
    /// <returns> result </returns>
    abstract float[] Run ( float[] inputs );

    /// <summary>
    /// run on data generator
    /// accuracy test
    /// </summary>
    /// <returns>accuracy %</returns>
    abstract float Run ( );
}