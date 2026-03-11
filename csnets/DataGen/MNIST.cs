namespace csnets.DataGen;

public struct MNISTImage {
    public float[] Pixels;  // 784 floats (28x28), normalized to [0, 1]
    public int Label;        // 0-9
    public float[] LabelOneHot; // 10 floats, one-hot encoded
}

public class MNIST {

    public string DataDir;

    public List<MNISTImage> TrainSet { get; private set; } = [];
    public List<MNISTImage> TestSet { get; private set; } = [];

    public MNIST ( string dataDir = "Data/MNIST" ) {
        DataDir = dataDir;
    }

    public void Load () {
        TrainSet = LoadImages (
            Path.Combine ( DataDir, "train-images-idx3-ubyte" ),
            Path.Combine ( DataDir, "train-labels-idx1-ubyte" )
        );

        TestSet = LoadImages (
            Path.Combine ( DataDir, "t10k-images-idx3-ubyte" ),
            Path.Combine ( DataDir, "t10k-labels-idx1-ubyte" )
        );
    }

    private static List<MNISTImage> LoadImages ( string imagesPath, string labelsPath ) {
        byte[] imageBytes = File.ReadAllBytes ( imagesPath );
        byte[] labelBytes = File.ReadAllBytes ( labelsPath );

        int imageMagic = ReadInt32BigEndian ( imageBytes, 0 );
        int numImages = ReadInt32BigEndian ( imageBytes, 4 );
        int rows = ReadInt32BigEndian ( imageBytes, 8 );
        int cols = ReadInt32BigEndian ( imageBytes, 12 );

        int labelMagic = ReadInt32BigEndian ( labelBytes, 0 );
        int numLabels = ReadInt32BigEndian ( labelBytes, 4 );

        if ( imageMagic != 2051 )
            throw new InvalidDataException ( $"Invalid image file magic number: {imageMagic}" );
        if ( labelMagic != 2049 )
            throw new InvalidDataException ( $"Invalid label file magic number: {labelMagic}" );
        if ( numImages != numLabels )
            throw new InvalidDataException ( $"Image count ({numImages}) != label count ({numLabels})" );

        int pixelCount = rows * cols;
        List<MNISTImage> images = new ( numImages );

        for ( int i = 0; i < numImages; i++ ) {
            float[] pixels = new float[pixelCount];
            int imageOffset = 16 + i * pixelCount;

            for ( int p = 0; p < pixelCount; p++ ) {
                pixels[p] = imageBytes[imageOffset + p] / 255f;
            }

            int label = labelBytes[8 + i];

            float[] oneHot = new float[10];
            oneHot[label] = 1f;

            images.Add ( new MNISTImage {
                Pixels = pixels,
                Label = label,
                LabelOneHot = oneHot
            } );
        }

        return images;
    }

    private static int ReadInt32BigEndian ( byte[] data, int offset ) {
        return ( data[offset] << 24 )
             | ( data[offset + 1] << 16 )
             | ( data[offset + 2] << 8 )
             | data[offset + 3];
    }

}
