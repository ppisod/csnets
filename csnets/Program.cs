using csnets.DataGen;
using csnets.Neural.Activations;
using csnets.Neural.Loss;
using csnets.Neural.Networks;

FeedForwardNet net = new ( 2, 3, 1, 3, 5 );

MidPointGenFunc gen = new ()
{
    c = 7,
    m = 1,
};
List<MidPointGenFuncOutput> dataset = gen.gen(10000);

foreach (var datapoint in dataset)
{
    net.Train <ReLU, MeanSquaredError> ([datapoint.in_A, datapoint.in_B ], [datapoint.out_M], 0.001f, true);
}