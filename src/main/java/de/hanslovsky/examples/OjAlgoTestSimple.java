package de.hanslovsky.examples;

import org.ojalgo.access.Access2D;
import org.ojalgo.array.Array1D;
import org.ojalgo.matrix.BasicMatrix.Builder;
import org.ojalgo.matrix.PrimitiveMatrix;
import org.ojalgo.matrix.decomposition.Eigenvalue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.MatrixStore.LogicalBuilder;
import org.ojalgo.scalar.ComplexNumber;

public class OjAlgoTestSimple
{

	public static void main( final String[] args )
	{

		final Builder< PrimitiveMatrix > b = PrimitiveMatrix.FACTORY.getBuilder( 3, 3 );
		for ( int i = 0; i < b.count(); ++i )
			b.set( i, i );
		final PrimitiveMatrix m = b.build();

		final LogicalBuilder< Double > w = MatrixStore.PRIMITIVE.makeWrapper( m );

		System.out.println( m );
		System.out.println( w );
		final Eigenvalue< Double > ev = Eigenvalue.PRIMITIVE.make();
//		ev.computeValuesOnly( m );
		final Access2D< Number > myM = new Access2D< Number >()
		{

			@Override
			public long countColumns()
			{
				return 3;
			}

			@Override
			public long countRows()
			{

				return 3;
			}

			@Override
			public long count()
			{
				return 9;
			}

			@Override
			public double doubleValue( final long index )
			{
				return index;
			}

			@Override
			public Number get( final long index )
			{
				return doubleValue( index );
			}

			@Override
			public double doubleValue( final long row, final long column )
			{
				return doubleValue( 3 * row + column );
			}

			@Override
			public Number get( final long row, final long column )
			{
				return doubleValue( row, column );
			}
		};

		final LogicalBuilder< Double > myW = MatrixStore.PRIMITIVE.makeWrapper( myM );

		ev.computeValuesOnly( myW );
		final Array1D< ComplexNumber > myEV = ev.getEigenvalues();
//		ev.compute( m, true );
		ev.computeValuesOnly( w );
		final Array1D< ComplexNumber > e = ev.getEigenvalues();
		System.out.println( myEV );
		System.out.println( e );


	}

}
