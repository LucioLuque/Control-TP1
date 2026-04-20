import math
import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

def search_zeta_wn(poles):
    print(f"{'zeta':>6} {'wn':>6} {'z':>8} {'Kp':>8} {'Ki':>8} {'ts_sim':>8} {'Mp_sim':>8}")
    print("-" * 65)

    for zeta in np.linspace(0.517, 0.85, 30):
        for wn in np.linspace(2.0, 6.0, 40):
            s0_re = -zeta * wn
            s0_im = wn * math.sqrt(1 - zeta**2)
            s0 = complex(s0_re, s0_im)
            
            sum_angles = 0
            for p in poles:
                angle = math.degrees(math.atan2(s0_im, s0_re - p))
                sum_angles -= angle
            
            angulo_f = -180 - sum_angles
            
            if not (5 < angulo_f < 175):
                continue
                
            z_val = s0_im / math.tan(math.radians(angulo_f)) - s0_re
            
            if z_val <= 0:
                continue
                
            num = s0**2 * (0.003*s0**2 + 0.0515*s0 + 0.2)
            den = 0.25 * (s0 + z_val)
            kp_val = abs(num / den)
            ki_val = kp_val * z_val
            
            try:
                CG = ctrl.TransferFunction(
                    [0.25*kp_val, 0.25*ki_val], 
                    [0.003, 0.0515, 0.2, 0, 0]
                )
                T = ctrl.feedback(CG, 1)
                info = ctrl.step_info(T, SettlingTimeThreshold=0.02)
                ts = info["SettlingTime"]
                mp = info["Overshoot"]
                
                if ts <= 7.5 and mp <= 15:
                    print(f"{zeta:>6.3f} {wn:>6.3f} {z_val:>8.4f} {kp_val:>8.4f} {ki_val:>8.4f} {ts:>8.3f} {mp:>8.3f}")
            except:
                continue

def angle_from_pole_to_s0(s0, pole_real):
    real_part = s0[0] - pole_real
    imag_part = s0[1]
    return math.degrees(math.atan2(imag_part, real_part))

def search_pi(zeta, wn, poles):
    s0 = [-zeta * wn, wn * math.sqrt(1 - zeta**2)]  # real, imaginary
    print(f"Chosen s0: {s0[0]} + j{s0[1]}")

    sum_angles = 0
    for pole_real in poles:
        angle = angle_from_pole_to_s0(s0, pole_real)
        sum_angles -= angle
        print(f"Pole: {pole_real}, Angle (degrees): {angle:.4f}")

    print(f"Sum of pole angles: {sum_angles:.4f} degrees")
    angulo_faltante = -180 - sum_angles

    
    z = s0[1] / math.tan(math.radians(angulo_faltante)) - s0[0]
    print(f"Zero válido en: z = {z:.4f}")

    s0 = complex(s0[0], s0[1])

    numerator = s0**2 * (0.003*s0**2 + 0.0515*s0 + 0.2)
    denominator = 0.25 * (s0 + z)

    Kp = abs(numerator / denominator)
    print(f"Kp = {Kp}")
    print(f"K_i = {Kp * z}")

    CG = ctrl.TransferFunction([0.25 * Kp, 0.25 * Kp * z], [0.003, 0.0515, 0.2, 0, 0])
    T = ctrl.feedback(CG, 1)

    info = ctrl.step_info(T, SettlingTimeThreshold=0.02)
    print("ts =", info["SettlingTime"])
    print("Mp =", info["Overshoot"])

    t = np.linspace(0, 20, 1000)
    t, y = ctrl.step_response(T, T=t)
    plt.figure(figsize=(10,6))
    plt.plot(t, y, label='Respuesta al escalón')
    plt.axhline(info["SteadyStateValue"], linestyle='--', label='Valor final')
    plt.axhline(1.02 * info["SteadyStateValue"], linestyle=':', label='Banda 2%')
    plt.axhline(0.98 * info["SteadyStateValue"], linestyle=':')
    plt.xlabel('Tiempo [s]', fontsize=12)
    plt.ylabel('Salida', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
    return z, Kp