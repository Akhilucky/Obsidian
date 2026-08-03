import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Locale;

/**
 * PortfolioOptimizer - pure-Java mean-variance optimization.
 *
 * Reads from stdin:
 *   Line 1: n (number of assets)
 *   Line 2: ticker names, comma separated
 *   Line 3: expected returns, comma separated
 *   Lines 4..4+n-1: covariance matrix rows, comma separated
 *
 * Writes to stdout (JSON):
 *   {"min_variance": {...}, "max_sharpe": {...}, "sharpe": ...}
 *
 * Uses closed-form solutions:
 *   w_minvar  = inv(C) * 1 / (1^T inv(C) 1)
 *   w_sharpe  = inv(C) * mu / (1^T inv(C) mu)
 */
public class PortfolioOptimizer {

    public static void main(String[] args) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        Locale.setDefault(Locale.US);

        int n = Integer.parseInt(in.readLine().trim());
        String[] tickers = in.readLine().trim().split("\\s*,\\s*");
        double[] mu = parseLine(in.readLine());

        double[][] cov = new double[n][n];
        for (int i = 0; i < n; i++) {
            cov[i] = parseLine(in.readLine());
        }

        if (tickers.length != n || mu.length != n) {
            System.out.println("{\"error\": \"dimension mismatch\"}");
            return;
        }

        double[][] inv = invert(cov, n);
        double[] one = new double[n];
        java.util.Arrays.fill(one, 1.0);

        double[] wMinVar = solve(inv, one, n);
        double[] wMaxSharpe = null;
        double sharpe = 0.0;

        double denom = dot(inv, mu, mu, n);
        if (Math.abs(denom) > 1e-12) {
            wMaxSharpe = solve(inv, mu, n);
            sharpe = weightedSharpe(wMaxSharpe, mu, cov, n);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"min_variance\": {");
        appendWeights(sb, tickers, wMinVar);
        sb.append("}, \"max_sharpe\": {");
        if (wMaxSharpe != null) {
            appendWeights(sb, tickers, wMaxSharpe);
        }
        sb.append("}, \"sharpe\": ").append(String.format("%.8f", sharpe)).append("}");
        System.out.println(sb.toString());
    }

    private static double[] parseLine(String line) {
        String[] parts = line.trim().split("\\s*,\\s*");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Double.parseDouble(parts[i].trim());
        return out;
    }

    private static double[] solve(double[][] invC, double[] vec, int n) {
        double[] w = matVec(invC, vec, n);
        double total = 0.0;
        for (int i = 0; i < n; i++) total += w[i];
        if (Math.abs(total) < 1e-12) return w;
        for (int i = 0; i < n; i++) w[i] /= total;
        return w;
    }

    private static double[] matVec(double[][] m, double[] v, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) s += m[i][j] * v[j];
            out[i] = s;
        }
        return out;
    }

    private static double dot(double[][] invC, double[] a, double[] b, int n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            double row = 0.0;
            for (int j = 0; j < n; j++) row += invC[i][j] * a[j];
            s += row * b[i];
        }
        return s;
    }

    private static double weightedSharpe(double[] w, double[] mu, double[][] cov, int n) {
        double ret = 0.0;
        for (int i = 0; i < n; i++) ret += w[i] * mu[i];
        double var = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) var += w[i] * w[j] * cov[i][j];
        }
        return ret / Math.sqrt(Math.max(var, 1e-12));
    }

    private static void appendWeights(StringBuilder sb, String[] tickers, double[] w) {
        for (int i = 0; i < tickers.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("\"").append(tickers[i]).append("\": ")
              .append(String.format("%.8f", w[i]));
        }
    }

    // Gauss-Jordan matrix inversion. Returns identity if singular.
    private static double[][] invert(double[][] a, int n) {
        double[][] m = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n + i] = 1.0;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            double best = Math.abs(m[col][col]);
            for (int r = col + 1; r < n; r++) {
                double v = Math.abs(m[r][col]);
                if (v > best) { best = v; pivot = r; }
            }
            if (best < 1e-12) continue;
            if (pivot != col) {
                double[] tmp = m[pivot]; m[pivot] = m[col]; m[col] = tmp;
            }
            double d = m[col][col];
            for (int j = 0; j < 2 * n; j++) m[col][j] /= d;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double f = m[r][col];
                for (int j = 0; j < 2 * n; j++) m[r][j] -= f * m[col][j];
            }
        }
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(m[i], n, inv[i], 0, n);
        return inv;
    }
}
