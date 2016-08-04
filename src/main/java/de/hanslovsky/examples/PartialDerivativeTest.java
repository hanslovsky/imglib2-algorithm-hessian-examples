package de.hanslovsky.examples;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ij.ImageJ;
import ij.ImagePlus;
import ij.plugin.FolderOpener;
import ij.process.ImageConverter;
import net.imglib2.FinalInterval;
import net.imglib2.algorithm.gradient.PartialDerivative;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.array.ArrayCursor;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;

public class PartialDerivativeTest
{
	public static void main( final String[] args ) throws IncompatibleTypeException, InterruptedException, ExecutionException
	{

//		final String url = "http://img.autobytel.com/car-reviews/autobytel/11694-good-looking-sports-cars/2016-Ford-Mustang-GT-burnout-red-tire-smoke.jpg";
		final String dir = "/data/hanslovskyp/davi_toy_set/substacks/scale=0.3/data";

		final ImagePlus imp = new FolderOpener().openFolder( dir );
		new ImageConverter( imp ).convertToGray32();

		final IntervalView< FloatType > wrapped = Views.interval( ImageJFunctions.wrapFloat( imp ), new FinalInterval( imp.getWidth(), imp.getHeight(), 10 ) );
		final long[] dim = new long[ wrapped.numDimensions() ];
		wrapped.dimensions( dim );

		final ArrayImg< FloatType, FloatArray > g = ArrayImgs.floats( dim );
		final ArrayImg< FloatType, FloatArray > g2 = ArrayImgs.floats( dim );
		final ArrayImg< FloatType, FloatArray > gp = ArrayImgs.floats( dim );

		final ExtendedRandomAccessibleInterval< FloatType, IntervalView< FloatType > > source = Views.extendBorder( wrapped );
		final int dimension = 0;

		final int nThreads = Runtime.getRuntime().availableProcessors();
		final ExecutorService es = Executors.newFixedThreadPool( nThreads );

		final int N = 100;

		long t = 0;
		for ( int i = 0; i < N; ++i )
		{
			final long t0 = System.currentTimeMillis();
			PartialDerivative.gradientCentralDifference( source, g, dimension );
			final long t1 = System.currentTimeMillis();
			t += t1 - t0;
		}
		System.out.println( "" + t / 1000.0 / N );

		long t2 = 0;
		for ( int i = 0; i < N; ++i )
		{
			final long t20 = System.currentTimeMillis();
			PartialDerivative.gradientCentralDifference2( source, g2, dimension );
			final long t21 = System.currentTimeMillis();
			t2 += t21 - t20;
		}
		System.out.println( "" + t2 / 1000.0 / N );

		long tp = 0;
		for ( int i = 0; i < N; ++i )
		{
			final long tp0 = System.currentTimeMillis();
			PartialDerivative.gradientCentralDifferenceParallel( source, gp, dimension, nThreads, es );
			final long tp1 = System.currentTimeMillis();
			tp += tp1 - tp0;
		}
		System.out.println( "" + tp / 1000.0 / N );

		for ( ArrayCursor< FloatType > c = g.cursor(), c2 = g2.cursor(), cp = gp.cursor(); c.hasNext(); )
		{
			final float v = c.next().get();
			final float v2 = c2.next().get();
			final float vp = cp.next().get();
			if ( v != v2 || v != vp || v2 != vp )
			{
				System.out.println( v + " " + v2 + " " + vp );
			}
		}

		System.out.println( "Done" );

		new ImageJ();
		ImageJFunctions.show( g, "g" );
		ImageJFunctions.show( g2, "g2" );
		ImageJFunctions.show( gp, "gp" );

	}
}
