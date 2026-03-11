namespace csnets.Neural.Activations;

public interface IActivation {
    static abstract float F ( float x );
    static abstract float Df ( float x );
}