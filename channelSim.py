from modules import *

# Meander channel simulation.
model = GeoModel(extent=[0, 1e4, 0, 1e4, 0, 1e3], resolution=[20.0, 20.0, 2.0])
model.add_rho_v(h_layer_range=[10, 20], vp_range=[3.5, 5.5], vs_range=[2.5, 3.1], rho_range=[2.2, 2.6])
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[750, 750],
                  W_range=[500, 500], D_range=[60, 60], n_bends_range=[20, 20], nit_range=[1500, 1500],
                  kl_range=[50, 50], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.1, 0.1], strike_range=[0, 0], h_lag_range=[25, 25], mode='complex',
                  save_it=20, h_levee_range=[0.85, 0.85], w_levee_range=[2000, 2000], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[650, 650],
                  W_range=[300, 300], D_range=[50, 50], n_bends_range=[20, 20], nit_range=[1500, 1500],
                  kl_range=[50, 50], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.1, 0.1], strike_range=[60, 60], h_lag_range=[20, 20], mode='complex',
                  save_it=20, h_levee_range=[0.8, 0.8], w_levee_range=[1200, 1200], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[550, 550],
                  W_range=[400, 400], D_range=[45, 45], n_bends_range=[20, 20], nit_range=[1500, 1500],
                  kl_range=[50, 50], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.08, 0.08], strike_range=[120, 120], h_lag_range=[18, 18], mode='complex',
                  save_it=20, h_levee_range=[0.8, 0.8], w_levee_range=[1600, 1600], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[450, 450],
                  W_range=[280, 280], D_range=[50, 50], n_bends_range=[20, 20], nit_range=[1800, 1800],
                  kl_range=[50, 50], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.06, 0.06], strike_range=[180, 180], h_lag_range=[20, 20], mode='complex',
                  save_it=20, h_levee_range=[0.75, 0.75], w_levee_range=[1120, 1120], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[350, 350],
                  W_range=[450, 450], D_range=[55, 55], n_bends_range=[20, 20], nit_range=[1600, 1600],
                  kl_range=[45, 45], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.12, 0.12], strike_range=[240, 240], h_lag_range=[25, 25], mode='complex',
                  save_it=20, h_levee_range=[0.85, 0.85], w_levee_range=[1800, 1800], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_meander(N=1, X_pos_range=None, Y_pos_range=None, Z_pos_range=[250, 250],
                  W_range=[350, 350], D_range=[60, 60], n_bends_range=[20, 20], nit_range=[2000, 2000],
                  kl_range=[60, 60], pad_down=0, delta_s=None, dt=0.1, perturb_range=None,
                  Cf_range=[0.1, 0.1], strike_range=None, h_lag_range=[25, 25], mode='complex',
                  save_it=20, h_levee_range=[0.8, 0.8], w_levee_range=[1400, 1400], show_migration=False,
                  figure_title='Meandering River Migration (execute cutoff)')  # Add meandering channel.
model.add_fold(N=15, sigma_range=[800, 1200], A_range=[40, 60], sync=True)  # Add folds.
model.add_fault(N=10, curved_fault_surface=False, reference_point_range=[500, 9500, 500, 9500, 300, 600],
                lx_range=[1.0, 1.0], ly_range=[0.06, 0.08], phi_range=None, theta_range=[50, 70],
                d_max_range=[0.03, 0.03], gamma_range=[0.04, 0.04], beta_range=[0.5, 0.5])
model.compute_impedance()
model.compute_rc()
model.rectangular_grid(resolution=[20, 20, 2], param=['lith_facies', 'vp', 'vs', 'density', 'rc'])
model.crop(param=['lith_facies', 'vp', 'vs', 'density', 'rc'])
model.make_synseis(plot_wavelet=False)
model.add_noise(noise_type='uniform', ratio=0.3)
np.save('lithology_model_real.npy', model.lith_facies)
np.save('vp_real.npy', model.vp)
np.save('vs_real.npy', model.vs)
np.save('density_real.npy', model.density)
np.save('rc_real.npy', model.rc)
np.save('seismic_real.npy', model.seis)
# Plot 3D model.
lith_cmap = ['gray', 'red', 'yellow', 'saddlebrown', 'blue']
p0 = BackgroundPlotter(shape=(1, 2))
p0.subplot(0, 0)
model.show(plotter=p0, param='lith_facies', zscale='auto', cmap=lith_cmap, slices=False)
p0.subplot(0, 1)
model.show(plotter=p0, param='seis', zscale='auto', cmap='seismic', slices=False)
p0.link_views()
p1 = BackgroundPlotter(shape=(1, 2))
p1.subplot(0, 0)
model.show(plotter=p1, param='lith_facies', zscale='auto', cmap=lith_cmap, slices=True)
p1.subplot(0, 1)
model.show(plotter=p1, param='seis', zscale='auto', cmap='seismic', slices=True)
p1.link_views()
p2 = BackgroundPlotter()
model.show(plotter=p2, param='lith_facies', zscale='auto', cmap=['red', 'yellow', 'saddlebrown', 'blue'],
           point_cloud=True, hide_value=[0])
p0.app.exec_()  # Show all figures.
p1.app.exec_()
p2.app.exec_()
