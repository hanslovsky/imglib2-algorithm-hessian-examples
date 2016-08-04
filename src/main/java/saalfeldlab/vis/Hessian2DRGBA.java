package saalfeldlab.vis;

import ij.ImageJ;
import ij.ImagePlus;
import net.imglib2.Cursor;
import net.imglib2.algorithm.corner.HessianMatrix;
import net.imglib2.algorithm.corner.TensorEigenValues;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;
import net.imglib2.view.composite.RealComposite;

public class Hessian2DRGBA
{
	public static void main( final String[] args ) throws IncompatibleTypeException
	{

		final ImageJ ij = new ImageJ();

		final String url = "http://img.autobytel.com/car-reviews/autobytel/11694-good-looking-sports-cars/2016-Ford-Mustang-GT-burnout-red-tire-smoke.jpg";
//		final String url = "http://mediad.publicbroadcasting.net/p/wuwm/files/styles/medium/public/201402/LeAnn_Crowe.jpg";

		final ImagePlus imp = new ImagePlus( url );
		imp.show();

		final Img< ARGBType > wrapped = ImageJFunctions.wrapRGBA( imp );

		final long[] dim = new long[ wrapped.numDimensions() + 1 ];
		for ( int d = 0; d < wrapped.numDimensions(); ++d )
		{
			dim[ d ] = wrapped.dimension( d );
		}
		dim[ dim.length - 1 ] = 3;

		final ArrayImg< DoubleType, DoubleArray > wrappedRGB = ArrayImgs.doubles( dim );
		final Cursor< ARGBType > s = wrapped.cursor();
		final Cursor< RealComposite< DoubleType > > t = Views.iterable( Views.collapseReal( wrappedRGB ) ).cursor();
		while ( s.hasNext() )
		{
			final int c = s.next().get();
			final RealComposite< DoubleType > d = t.next();
			d.get( 0 ).set( ARGBType.red( c ) );
			d.get( 1 ).set( ARGBType.green( c ) );
			d.get( 2 ).set( ARGBType.blue( c ) );
		}

//		ImageJFunctions.show( wrappedRGB );

		final double sigma = 2.0;

		final long[] gaussianDim = new long[] { imp.getWidth(), imp.getHeight(), 3 };
		final long[] gradientDim = new long[] { imp.getWidth(), imp.getHeight(), 2, 3 };
		final long[] hessianDim = new long[] { imp.getWidth(), imp.getHeight(), 3, 3 };
		final long[] evDim = new long[] { imp.getWidth(), imp.getHeight(), 2, 3 };

		final ArrayImg< DoubleType, DoubleArray > gaussians = ArrayImgs.doubles( gaussianDim );
		final ArrayImg< DoubleType, DoubleArray > gradients = ArrayImgs.doubles( gradientDim );
		final ArrayImg< DoubleType, DoubleArray > hessians = ArrayImgs.doubles( hessianDim );
		final ArrayImg< DoubleType, DoubleArray > eigenvals = ArrayImgs.doubles( evDim );

		for ( int i = 0; i < 3; ++i )
		{
			HessianMatrix.calculateMatrix(
					Views.extendBorder( Views.hyperSlice( wrappedRGB, 2, i ) ),
					Views.hyperSlice( gaussians, 2, i ),
					Views.hyperSlice( gradients, 3, i ),
					Views.hyperSlice( hessians, 3, i ),
					sigma,
					new OutOfBoundsBorderFactory<>()
					);
			System.out.println( "Done with hessian" );
			TensorEigenValues.calculateEigenValuesSymmetric(
					Views.hyperSlice( hessians, 3, i ),
					Views.hyperSlice( eigenvals, 3, i ) );
			System.out.println( "Done with evs" );
		}

		for ( int i = 0; i < 2; ++i )
		{
			final ImagePlus resultImp = ImageJFunctions.wrap( Views.hyperSlice( eigenvals, 2, i ), "" );

			resultImp.show();
		}
	}
}
