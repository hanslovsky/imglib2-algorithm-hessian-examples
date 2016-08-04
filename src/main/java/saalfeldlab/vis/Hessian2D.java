package saalfeldlab.vis;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ImageConverter;
import net.imglib2.algorithm.corner.HessianMatrix;
import net.imglib2.algorithm.corner.TensorEigenValues;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.outofbounds.OutOfBoundsBorderFactory;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class Hessian2D
{
	public static void main( final String[] args ) throws IncompatibleTypeException
	{

		final ImageJ ij = new ImageJ();

		final String dir = "/home/hanslovskyp/local/tmp/hessian-test";
//		final String url = "http://img.autobytel.com/car-reviews/autobytel/11694-good-looking-sports-cars/2016-Ford-Mustang-GT-burnout-red-tire-smoke.jpg";
		final String url = String.format( "file://%s/zero-one-range.tif", dir );

		final ImagePlus imp = new ImagePlus( url );
		new ImageConverter( imp ).convertToGray32();
		imp.show();

		final Img< FloatType > wrapped = ImageJFunctions.wrapFloat( imp );

		final double scale = 2.0;

		final int nThreads = Runtime.getRuntime().availableProcessors();
		final ExecutorService es = Executors.newFixedThreadPool( nThreads );
		final Img< DoubleType > hessian = HessianMatrix.calculateMatrix(
				Views.extendBorder( wrapped ),
				wrapped,
				scale,
				new OutOfBoundsBorderFactory<>(),
				new ArrayImgFactory<>(),
				new DoubleType(),
				nThreads,
				es );

		final int N = 10;
		long tHessian = 0;
		for ( int i = 0; i < N; ++i )
		{
			final long t0 = System.currentTimeMillis();
			HessianMatrix.calculateMatrix(
					Views.extendBorder( wrapped ),
					wrapped,
					scale,
					new OutOfBoundsBorderFactory<>(),
					new ArrayImgFactory<>(),
					new DoubleType(),
					nThreads,
					es );
			final long t1 = System.currentTimeMillis();
			tHessian += t1 - t0;
		}

		final double tHessianDouble = tHessian / 1000.0 / N;
		System.out.println( "tHessian=" + tHessianDouble + "s" );

		final Img< DoubleType > evs = TensorEigenValues.calculateEigenValuesSymmetric( hessian, new ArrayImgFactory< DoubleType >(), new DoubleType(), nThreads, es );

		long tEigenvals = 0;
		for ( int i = 0; i < N; ++i )
		{
			final long t0 = System.currentTimeMillis();
			TensorEigenValues.calculateEigenValuesSymmetric( hessian, new ArrayImgFactory< DoubleType >(), new DoubleType(), nThreads, es );
			final long t1 = System.currentTimeMillis();
			tEigenvals += t1 - t0;
		}
		final double tEigenvalsDouble = tEigenvals / 1000.0 / N;
		System.out.println( "tEigenvals=" + tEigenvalsDouble + "s" );
		es.shutdown();

//		new FileSaver( ImageJFunctions.wrap( Views.hyperSlice( hessian, 2, 0 ), "" ) ).saveAsTiff( dir + "/h-1.tif" );
//		new FileSaver( ImageJFunctions.wrap( Views.hyperSlice( hessian, 2, 1 ), "" ) ).saveAsTiff( dir + "/h-2.tif" );
//		new FileSaver( ImageJFunctions.wrap( Views.hyperSlice( hessian, 2, 2 ), "" ) ).saveAsTiff( dir + "/h-3.tif" );
//
//		new FileSaver( ImageJFunctions.wrap( Views.hyperSlice( evs, 2, 0 ), "" ) ).saveAsTiff( dir + "/e-1.tif" );
//		new FileSaver( ImageJFunctions.wrap( Views.hyperSlice( evs, 2, 1 ), "" ) ).saveAsTiff( dir + "/e-2.tif" );

		ImageJFunctions.show( hessian );
		ImageJFunctions.show( evs );
//
		final ArrayImg< DoubleType, DoubleArray > square = ArrayImgs.doubles( 5, 5 );
		final ArrayRandomAccess< DoubleType > ra = square.randomAccess();
		ra.setPosition( new long[] { 2, 2 } );
		ra.get().set( 4 );

		final Img< DoubleType > hs = HessianMatrix.calculateMatrix(
				Views.extendBorder( square ),
				square,
				0.1,
				new OutOfBoundsBorderFactory<>(),
				new ArrayImgFactory< DoubleType >(),
				new DoubleType() );
		final Img< DoubleType > evs2 = TensorEigenValues.calculateEigenValuesSymmetric( hs, new ArrayImgFactory< DoubleType >(), new DoubleType() );
		ImageJFunctions.show( evs2 );

	}
}
