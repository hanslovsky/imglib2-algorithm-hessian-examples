package de.hanslovsky.examples;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ImageConverter;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gradient.HessianMatrix;
import net.imglib2.algorithm.linalg.eigen.EigenValues2DSymmetric;
import net.imglib2.algorithm.linalg.eigen.EigenValuesSymmetric;
import net.imglib2.algorithm.linalg.eigen.TensorEigenValues;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealDoubleConverter;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class Hessian2D
{
	public static void main( final String[] args ) throws IncompatibleTypeException, InterruptedException, ExecutionException
	{

		final ImageJ ij = new ImageJ();

//		final String dir = "/home/hanslovskyp/local/tmp/hessian-test";
		final String url = "http://img.autobytel.com/car-reviews/autobytel/11694-good-looking-sports-cars/2016-Ford-Mustang-GT-burnout-red-tire-smoke.jpg";
//		final String url = String.format( "file://%s/zero-one-range.tif", dir );

		final ImagePlus imp = new ImagePlus( url );
		new ImageConverter( imp ).convertToGray32();
		imp.show();

		final RandomAccessibleInterval< DoubleType > wrapped = Converters.convert( ( RandomAccessibleInterval< FloatType > ) ImageJFunctions.wrapFloat( imp ), new RealDoubleConverter<>(), new DoubleType() );

		final double sigma = 2.0;

		final int nThreads = Runtime.getRuntime().availableProcessors();
		final ExecutorService es = Executors.newFixedThreadPool( nThreads );

		final ArrayImg< DoubleType, DoubleArray > hessian = ArrayImgs.doubles( imp.getWidth(), imp.getHeight(), 3 );

		{
			final ArrayImg< DoubleType, DoubleArray > gaussian = ArrayImgs.doubles( Intervals.dimensionsAsLongArray( wrapped ) );
			final ArrayImg< DoubleType, DoubleArray > gradients = ArrayImgs.doubles( imp.getWidth(), imp.getHeight(), 2 );
			HessianMatrix.calculateMatrix( Views.extendBorder( wrapped ), gaussian, gradients, hessian, new OutOfBoundsBorderFactory<>(), nThreads, es, sigma );
		}

		final int N = 100;
		final int nWarmup = 100;
		final int nTotal = N + nWarmup;

		final Img< DoubleType > evs1 = TensorEigenValues.createAppropriateResultImg( hessian, new ArrayImgFactory<>(), new DoubleType() );
		final Img< DoubleType > evs2 = TensorEigenValues.createAppropriateResultImg( hessian, new ArrayImgFactory<>(), new DoubleType() );

		{
			long tEigenvals = 0;
			for ( int i = 0; i < nTotal; ++i )
			{
				final long t0 = System.currentTimeMillis();
				TensorEigenValues.calculateEigenValues( hessian, evs1, new EigenValues2DSymmetric<>(), nThreads, es );
				final long t1 = System.currentTimeMillis();
				if ( i > nWarmup )
					tEigenvals += t1 - t0;
			}
			final double tEigenvalsDouble = tEigenvals / 1000.0 / N;
			System.out.println( "tEigenvals=" + tEigenvalsDouble + "s" );
		}

		{
			long tEigenvals = 0;
			for ( int i = 0; i < nTotal; ++i )
			{
				final long t0 = System.currentTimeMillis();
				TensorEigenValues.calculateEigenValues( hessian, evs2, new EigenValuesSymmetric<>( 2 ), nThreads, es );
				final long t1 = System.currentTimeMillis();
				if ( i > nWarmup )
					tEigenvals += t1 - t0;
			}
			final double tEigenvalsDouble = tEigenvals / 1000.0 / N;
			System.out.println( "tEigenvals(oj)=" + tEigenvalsDouble + "s" );
		}

		es.shutdown();

		ImageJFunctions.show( hessian );
		ImageJFunctions.show( evs1 );
		ImageJFunctions.show( evs2 );

	}
}
