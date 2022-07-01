from modules import *

random.seed(0)
np.random.seed(0)

test = 'test2'

if test == 'test1':
    # Folds and faults simulation.
    model = GeoModel(extent=[0, 1e4, 0, 1e4, 0, 2e3], resolution=[100, 100, 20])
    model.add_rho_v(h_layer_range=[20, 100], vp_range=[3.0, 5.5], rho_range=[2.3, 2.7])
    model.compute_impedance()
    model.compute_rc()
    model.add_fold()
    model.add_fault(N=3, mark_faults=True, curved_fault_surface=False, computation_mode='parallel',
                    reference_point_range=[0, 1e4, 0, 1e4, 500, 1500],
                    phi_range=[90, 90], theta_range=[70, 80], lx_range=[1.0, 1.0], ly_range=[0.2, 0.3],
                    gamma_range=[0.1, 0.1], beta_range=[0.5, 0.5], d_max_range=[0.1, 0.1])
    model.make_synseis(plot_wavelet=False)
    model.add_noise(noise_type='uniform', ratio=0.05)
    print(np.amin(model.seis), np.amax(model.seis))
    # Plot 3D models.
    pv.set_plot_theme('ParaView')  # Set plot theme as ParaView.
    p0 = BackgroundPlotter(shape=(2, 2))
    p0.subplot(0, 0)  # P-wave velocity model.
    model.show(plotter=p0, param='vp', cmap='rainbow', zscale='auto')
    p0.subplot(0, 1)  # Density model.
    model.show(plotter=p0, param='density', cmap='rainbow', zscale='auto')
    p0.subplot(1, 0)  # Reflection coefficient model.
    model.show(plotter=p0, param='rc', cmap='gray_r', zscale='auto')
    p0.subplot(1, 1)  # Fault probability model.
    model.show(plotter=p0, param='seis', cmap='seismic', zscale='auto')
    p0.link_views()
    # Resample on grid.
    model.rectangular_grid(param=['rc', 'vp', 'density'], resolution=[20, 20, 4])
    model.crop(param=['rc', 'vp', 'density'])
    model.make_synseis(plot_wavelet=False)
    np.save('rc.npy', model.rc)
    np.save('vp.npy', model.vp)
    np.save('density.npy', model.density)
    np.save('seis.npy', model.seis)
    # Plot 3D models.
    p1 = BackgroundPlotter(shape=(2, 2))
    p1.subplot(0, 0)
    model.show(plotter=p1, param='vp', cmap='rainbow', zscale='auto')
    p1.subplot(0, 1)
    model.show(plotter=p1, param='density', cmap='rainbow', zscale='auto')
    p1.subplot(1, 0)
    model.show(plotter=p1, param='rc', cmap='gray_r', zscale='auto')
    p1.subplot(1, 1)
    model.show(plotter=p1, param='seis', cmap='seismic', zscale='auto')
    p1.link_views()
    p0.app.exec_()
    p1.app.exec_()  # Show all figures.
