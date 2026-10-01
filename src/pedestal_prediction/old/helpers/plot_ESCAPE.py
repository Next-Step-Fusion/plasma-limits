    if te_plot_profiles:
        red_blue = LinearSegmentedColormap.from_list('red_blue', ['red', 'blue'])
        te_colors = red_blue(np.linspace(0, 1, len(te_plot_profiles)))
        ne_colors = red_blue(np.linspace(0, 1, len(ne_plot_profiles)))

        fig, ax = plt.subplots(figsize=(6, 4))
        for prof, color in zip(te_plot_profiles, te_colors):
            ax.plot(prof['psi_N'], prof['y'] / 1e3, lw=2, ls=prof['ls'],
                    color=color, label=prof['label'])
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$T_e$ [keV]')
        ax.set_title('Solved $T_e$ profiles')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(psi_N_inner, 1.0)
        fig.tight_layout()

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        for prof, color in zip(ne_plot_profiles, ne_colors):
            ax1.plot(prof['psi_N'], prof['y'], lw=2, ls=prof['ls'],
                     color=color, label=prof['label'])
        ax1.set_xlabel(r'$\psi_N$')
        ax1.set_ylabel(r'$n_e$ ($10^{19}$ m$^{-3}$)')
        ax1.set_title('Solved $n_e$ profiles')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_xlim(psi_N_inner, 1.0)
        fig1.tight_layout()
        plt.show()