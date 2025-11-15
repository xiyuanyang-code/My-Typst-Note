public class Example {
	public static void main(String[] args) {
		System.out.println("Hello World!");
	}

	public static boolean example(int a, int b) {
		int x = a + b - 100;
		int y = a*b;
		if (x > y) {
			System.out.print("loc1");
			if (b > 10) {
				if (x == 100001) {
					System.out.print("loc2");
					return false;
				} else {
					System.out.print("loc3");
					return true;
				}
			}
		}
		
		return true;
	}
}