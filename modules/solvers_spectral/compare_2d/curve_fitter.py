import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def curve_fitter(x_list, y_list, comparison_type, yaxislabel):


    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize = (14,7)
    )

    # regular L2 norm
    ax1.plot(x_list, y_list)
    ax1.set_xlabel(r"$\alpha$ value")
    ax1.set_ylabel(f"{yaxislabel}")
    ax1.grid()

    if comparison_type == "power":

        # log-log of L2 norm w/ fitting line
        fit_slope, fit_intercept = np.polyfit(np.log(x_list), np.log(y_list), 1)
        fit_line = np.exp(fit_intercept) * x_list**fit_slope

        # power fit (a* x^b)
        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, 'b-', label='Fitted line', color='orange')
        ax2.set_xlabel(r"log($\alpha$ value)")
        ax2.set_ylabel(f"log(yaxislabel)")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01, f"Fitted line = {np.exp(fit_intercept):.2e} x^{fit_slope:.2e}", ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "exp":

        # remove undefined terms
        mask = y_list > 0
        x = x_list[mask]
        y = y_list[mask]
        log_y = np.log(y)

        # fit linear: log_y = b*x + log(a)
        b, log_a = np.polyfit(x, log_y, 1)
        a = np.exp(log_a)
        fit_line = a * np.exp(b * x_list)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, 'b-', label='Fitted line', color='orange')
        ax2.set_xlabel(r"log($\alpha$ value)")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01, f"Fitted line = {a:.2e} e^{b:.2e}x", ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "iter":

        def nested_exp(x, a, b, c):
            return a * np.exp(b * np.exp(c * x))

        # remove invalid values
        mask = y_list > 0
        x = x_list[mask]
        y = y_list[mask]

        # initial guess
        a0 = np.min(y)
        b0 = -0.1
        c0 = -0.1
        p0 = [a0, b0, c0]

        # fit
        params, cov = curve_fit(nested_exp, x, y, p0=p0, maxfev=50000, 
                                bounds=([0, -10, -10], [np.inf, 10, 10]) )
        a, b, c = params
        fit_line = nested_exp(x_list, a, b, c)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"log($\alpha$ value)")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01, f"Fitted line = {a:.2e} e^({b:.2e} e^{c:.2e}x)", ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "sat_exp":

        def sat_exp(x, A, b):
            return A * (1 - np.exp(-b * x))

        # remove invalid values
        mask = y_list > 0
        x = x_list[mask]
        y = y_list[mask]

        # initial guess
        A0 = np.max(y)
        b0 = 0.1
        p0 = [A0, b0]

        # fit
        params, cov = curve_fit(sat_exp, x, y, p0=p0, maxfev=50000)
        A, b = params
        fit_line = sat_exp(x_list, A, b)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"log($\alpha$ value)")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01,
            f"Fitted line = {A:.2e} (1 - e^(-{b:.2e} x))",
            ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "log":

        def log_model(x, A, b):
            return A * np.log(1 + b * x)

        # remove invalid values
        mask = (y_list > 0) & (x_list > 0)
        x = x_list[mask]
        y = y_list[mask]

        # initial guess
        A0 = np.max(y)
        b0 = 0.1
        p0 = [A0, b0]

        # fit
        params, cov = curve_fit(log_model, x, y, p0=p0, maxfev=50000)
        A, b = params
        fit_line = log_model(x_list, A, b)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"log($\alpha$ value)")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01,
            f"Fitted line = {A:.2e} log(1 + {b:.2e} x)",
            ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "log_sat":

        def log_sat(x, A, b):
            return A * np.log(1 + b * x) / np.log(1 + b)

        mask = (y_list > 0) & (x_list > 0)
        x = x_list[mask]
        y = y_list[mask]

        A0 = np.max(y)
        b0 = 0.1
        p0 = [A0, b0]

        params, cov = curve_fit(log_sat, x, y, p0=p0, maxfev=50000)
        A, b = params
        fit_line = log_sat(x_list, A, b)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01,
            f"Fitted line = {A:.2e} log(1 + {b:.2e} x)/log(1+{b:.2e})",
            ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "log_power":

        def log_power(x, A, b, c):
            return A * (np.log(1 + b * x))**c

        mask = (y_list > 0) & (x_list > 0)
        x = x_list[mask]
        y = y_list[mask]

        A0 = np.max(y)
        b0 = 0.1
        c0 = 1.0
        p0 = [A0, b0, c0]

        params, cov = curve_fit(log_power, x, y, p0=p0, maxfev=50000)
        A, b, c = params
        fit_line = log_power(x_list, A, b, c)

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01,
            f"{A:.2e} [log(1 + {b:.2e} x)]^{c:.2e}",
            ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    elif comparison_type == "logistic":

        def rational_power(x, A, B, p):
            return A * ( (x / B)**p ) / (1 + (x / B)**p)

        mask = (y_list >= 0) & (x_list > 0)
        x = x_list[mask]
        y = y_list[mask]

        A0 = np.max(y)
        B0 = np.median(x)
        p0 = 5.0   # increase this if corner isn't sharp enough

        p0 = [A0, B0, p0]

        params, _ = curve_fit(
            rational_power, x, y,
            p0=p0,
            maxfev=100000,
            bounds=([0, 0, 1], [np.inf, np.inf, 50])
        )

        A, B, p = params
        fit_line = rational_power(x_list, A, B, p)

        print(f"A={A}")
        print(f"B={B}")
        print(f"p={p}")

        ax2.loglog(x_list, y_list, 'o', label='Data')
        ax2.loglog(x_list, fit_line, '-', label='Fitted line', color='orange')

        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(f"log({yaxislabel})")
        ax2.grid()
        ax2.legend()

        plt.figtext(0.5, 0.01,
            f"{A:.2e} [(x/{B:.2e})^{p:.2e} / (1 + (x/{B:.2e})^{p:.2e})]",
            ha='center', fontsize=10, color='gray')
        plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    plt.savefig("compare_spectral_NSVvsNSE.png", dpi=200, bbox_inches='tight')
    #plt.show()