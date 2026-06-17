import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# paper Fig.2 scaling: tau=t*sqrt(g/R), V=v/sqrt(Rg), NF=N/(mg)
g = 1.0; R = 1.0; m = 1.0; eps = 1e-12


def eq(s, mu):
    theta, theta_v, phi, phi_v = s
    sinT = math.sin(theta); cosT = math.cos(theta)
    if abs(sinT) < eps: sinT = eps if sinT >= 0 else -eps

    D = math.sqrt(theta_v**2 + phi_v**2 * sinT**2)
    if D < eps: return np.array([theta_v, 0.0, phi_v, 0.0])

    theta_a = phi_v**2 * sinT * cosT + (g/R) * sinT - (mu*g*theta_v*cosT)/(R*D) + mu*theta_v*D
    phi_a = -2*theta_v*phi_v*cosT/sinT - (mu*g*phi_v*cosT)/(R*D) + mu*phi_v*D
    return np.array([theta_v, theta_a, phi_v, phi_a])


def rk4(f, s, dt, *args):
    k1 = f(s, *args); k2 = f(s + dt*k1/2, *args)
    k3 = f(s + dt*k2/2, *args); k4 = f(s + dt*k3, *args)
    return s + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def vals(s):
    theta, theta_v, phi, phi_v = s
    D = math.sqrt(theta_v**2 + phi_v**2 * math.sin(theta)**2)
    v = R * D; V = v / math.sqrt(R*g)
    N = m * (g * math.cos(theta) - R * D**2); NF = N / (m*g)
    x = R * math.sin(theta) * math.cos(phi); y = R * math.sin(theta) * math.sin(phi); z = R * math.cos(theta)
    return dict(D=D, v=v, V=V, N=N, NF=NF, x=x, y=y, z=z)


def row(t, s, mu):
    theta, theta_v, phi, phi_v = s; d = vals(s)
    return dict(t=t, tau=t*math.sqrt(g/R), theta=theta, theta_deg=math.degrees(theta), theta_v=theta_v,
                phi=phi, phi_deg=math.degrees(phi), phi_v=phi_v, mu=mu, **d)


def cut(s0, s1, y0, y1, target=0.0):
    a = 1.0 if abs(y1-y0) < eps else (target-y0)/(y1-y0)
    a = max(0.0, min(1.0, a))
    return s0 + a*(s1-s0), a


def init_state(theta0=10.0, phi0=0.0, V0=0.5, angle0=45.0):
    theta = math.radians(theta0); phi = math.radians(phi0); a = math.radians(angle0)
    v0 = V0 * math.sqrt(R*g)
    theta_v = (v0/R) * math.cos(a); phi_v = (v0/(R*math.sin(theta))) * math.sin(a)
    return np.array([theta, theta_v, phi, phi_v], dtype=float)


def run_slide(mu, dt=1e-4, tMax=20.0, stopV=1e-4, minStopT=0.05, **start):
    t = 0.0; s = init_state(**start)
    rowL = [row(t, s, mu)]; why = 'tMax'

    for _ in range(math.ceil(tMax/dt)):
        oldS = s; oldT = t; oldN = vals(oldS)['N']
        newS = rk4(eq, oldS, dt, mu); newT = oldT + dt
        newD = vals(newS); newV = newD['V']; newN = newD['N']

        if newN <= 0:
            endS, a = cut(oldS, newS, oldN, newN); rowL.append(row(oldT + a*dt, endS, mu))
            why = 'detached'; break

        if newS[0] >= math.pi/2:
            endS, a = cut(oldS, newS, oldS[0], newS[0], math.pi/2); rowL.append(row(oldT + a*dt, endS, mu))
            why = 'equator'; break

        if newT > minStopT and newV <= stopV:
            endS = newS.copy(); endS[1] = 0.0; endS[3] = 0.0
            rowL.append(row(newT, endS, mu)); why = 'stopped'; break

        t = newT; s = newS; rowL.append(row(t, s, mu))

    df = pd.DataFrame(rowL); df['why'] = why
    return df


def run_all(muL=(0.0, 0.3, 3.0), **start):
    dfL = [run_slide(mu, **start) for mu in muL]
    df = pd.concat(dfL, ignore_index=True)

    res = pd.DataFrame(dict(mu=mu, why=end['why'], tau=end['tau'], theta_deg=end['theta_deg'],
                            phi_deg=end['phi_deg'], V=end['V'], NF=end['NF'])
                       for mu in muL for end in [df[df['mu'] == mu].iloc[-1]])

    return df, res


def plot_fig2(df):
    fig = make_subplots(rows=2, cols=2, subplot_titles=('theta(t)', 'phi(theta)', 'NF(theta)', 'V(theta)'))

    for mu, d in ((mu, df[df['mu'] == mu]) for mu in sorted(df['mu'].unique())):
        name = 'mu={:g}'.format(mu)
        fig.add_trace(go.Scatter(x=d['tau'], y=d['theta_deg'], mode='lines', name=name), 1, 1)
        fig.add_trace(go.Scatter(x=d['theta_deg'], y=d['phi_deg'], mode='lines', showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=d['theta_deg'], y=d['NF'], mode='lines', showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=d['theta_deg'], y=d['V'], mode='lines', showlegend=False), 2, 2)

    xLabel = ('tau = t sqrt(g/R)', 'theta [deg]', 'theta [deg]', 'theta [deg]')
    yLabel = ('theta [deg]', 'phi [deg]', 'NF = N/(mg)', 'V = v/sqrt(Rg)')

    for (r, c), xT, yT in zip(((1,1), (1,2), (2,1), (2,2)), xLabel, yLabel):
        fig.update_xaxes(title_text=xT, row=r, col=c); fig.update_yaxes(title_text=yT, row=r, col=c)

    fig.update_layout(title='Case B RK4 - Fig.2 style', width=1100, height=800)
    return fig


def plot_3d(df):
    fig = go.Figure()
    u = np.linspace(0, 2*np.pi, 70); v = np.linspace(0, np.pi/2, 35)
    X = R * np.outer(np.cos(u), np.sin(v)); Y = R * np.outer(np.sin(u), np.sin(v)); Z = R * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.25, showscale=False, name='sphere'))

    for mu, d in ((mu, df[df['mu'] == mu]) for mu in sorted(df['mu'].unique())):
        fig.add_trace(go.Scatter3d(x=d['x'], y=d['y'], z=d['z'], mode='lines', name='mu={:g}'.format(mu)))
        fig.add_trace(go.Scatter3d(x=[d['x'].iloc[-1]], y=[d['y'].iloc[-1]], z=[d['z'].iloc[-1]],
                                   mode='markers', name='end {:g}'.format(mu)))

    fig.update_layout(title='3D path on sphere', width=900, height=800,
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'))
    return fig


def main():
    df, res = run_all()
    print(res.to_string(index=False))

    fig = plot_fig2(df); fig.show()
    fig3d = plot_3d(df); fig3d.show()

    # fig.write_html('caseB_fig2.html'); fig3d.write_html('caseB_3d.html')


if __name__ == '__main__':
    main()