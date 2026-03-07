import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

G = 6.67430e-11

class PointMass:
    def __init__(self, Position, Mass):
        self.Position = np.array(Position, dtype=float)
        self.Mass = float(Mass)

class DiscretePlanet:
    def __init__(self, PlanetRadius=1000.0, CoreRadius=400.0):
        self.PlanetRadius = PlanetRadius
        self.CoreRadius = CoreRadius
        self.Points = []
    def _random_points_in_sphere(self, Count, Radius, RandState):
        u = RandState.rand(Count)
        cos_theta = 2*RandState.rand(Count)-1
        phi = 2*np.pi*RandState.rand(Count)
        r = Radius * np.cbrt(u)
        sin_theta = np.sqrt(1-cos_theta**2)
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * cos_theta
        return np.vstack((x,y,z)).T
    def Populate(self, NMantlePoints=5000, MantleMassTotal=5.9e24, KCorePoints=2000, CoreMassTotal=1.0e24, Seed=0):
        RandState = np.random.RandomState(Seed)
        self.Points = []
        MantleRadius = self.PlanetRadius
        inner = self.CoreRadius
        PositionsMantle = self._random_points_in_sphere(NMantlePoints, MantleRadius, RandState)
        dists = np.linalg.norm(PositionsMantle, axis=1)
        maskMantle = dists >= inner
        PositionsMantle = PositionsMantle[maskMantle]
        MantleMassPerPoint = MantleMassTotal / max(1, PositionsMantle.shape[0])
        for pos in PositionsMantle:
            self.Points.append(PointMass(pos, MantleMassPerPoint))
        PositionsCore = self._random_points_in_sphere(KCorePoints, self.CoreRadius, RandState)
        CoreMassPerPoint = CoreMassTotal / max(1, KCorePoints)
        for pos in PositionsCore:
            self.Points.append(PointMass(pos, CoreMassPerPoint))
    def ComputeGravityAt(self, Position, MinDistance=1.0, ExclusionRadius=0.0):
        Pos = np.array(Position, dtype=float)
        if len(self.Points) == 0:
            return np.array([0.0, 0.0, 0.0])
        AllPos = np.array([p.Position for p in self.Points])
        AllMass = np.array([p.Mass for p in self.Points])
        Rvec = AllPos - Pos
        Dist = np.sqrt(np.sum(Rvec**2, axis=1))
        if ExclusionRadius > 0.0:
            mask = Dist >= ExclusionRadius
            if not np.any(mask):
                return np.array([0.0, 0.0, 0.0])
            AllPos = AllPos[mask]
            AllMass = AllMass[mask]
            Rvec = Rvec[mask]
            Dist = Dist[mask]
        Eps = 1e-300
        DistSafe = np.maximum(Dist, MinDistance)
        UnitVec = np.where(
            (Dist.reshape(-1, 1) > Eps),
            Rvec / (Dist.reshape(-1, 1) + Eps),
            0.0
        )
        contribs = (G * AllMass[:, np.newaxis]) * UnitVec / (DistSafe**2).reshape(-1, 1)
        Total = np.sum(contribs, axis=0)
        return np.asarray(Total, dtype=float)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.Figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.Figure)
        self.Ax = self.Figure.add_subplot(111)

class MplCanvas3D(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.Figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.Figure)
        self.Ax = self.Figure.add_subplot(111, projection='3d')

class GravitySimulatorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Discrete Gravity Simulator")
        self.Planet = DiscretePlanet()
        self._BuildInterface()
        self._ConnectSignals()
        self.Seed = 0
    def _BuildInterface(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout()
        central.setLayout(layout)
        controls = QtWidgets.QWidget()
        controlsLayout = QtWidgets.QFormLayout()
        controls.setLayout(controlsLayout)
        self.SpinNMantle = QtWidgets.QSpinBox()
        self.SpinNMantle.setRange(0,2000000)
        self.SpinNMantle.setValue(6000)
        self.SpinKCore = QtWidgets.QSpinBox()
        self.SpinKCore.setRange(0,2000000)
        self.SpinKCore.setValue(3000)
        self.EditMantleMass = QtWidgets.QLineEdit("5.0")
        self.EditCoreMass = QtWidgets.QLineEdit("10.0")
        self.EditPlanetRadius = QtWidgets.QLineEdit(str(self.Planet.PlanetRadius))
        self.EditCoreRadius = QtWidgets.QLineEdit(str(self.Planet.CoreRadius))
        self.ButtonGenerate = QtWidgets.QPushButton("Generate Planet")
        self.LabelStatus = QtWidgets.QLabel("Ready")
        controlsLayout.addRow("Mantle points N:", self.SpinNMantle)
        controlsLayout.addRow("Core points K:", self.SpinKCore)
        controlsLayout.addRow("Mantle mass (kg):", self.EditMantleMass)
        controlsLayout.addRow("Core mass (kg):", self.EditCoreMass)
        controlsLayout.addRow("Planet radius (m):", self.EditPlanetRadius)
        controlsLayout.addRow("Core radius (m):", self.EditCoreRadius)
        controlsLayout.addRow(self.ButtonGenerate)
        controlsLayout.addRow(self.LabelStatus)
        playerGroup = QtWidgets.QGroupBox("Player / Query")
        playerLayout = QtWidgets.QFormLayout()
        playerGroup.setLayout(playerLayout)
        self.EditPlayerX = QtWidgets.QLineEdit("0.0")
        self.EditPlayerY = QtWidgets.QLineEdit("0.0")
        self.EditPlayerZ = QtWidgets.QLineEdit(str(self.Planet.PlanetRadius))
        self.ButtonSetSurface = QtWidgets.QPushButton("Set to Surface (+Z)")
        self.ButtonRandom = QtWidgets.QPushButton("Random Position")
        self.ButtonCompute = QtWidgets.QPushButton("Analyze Gravity at Point")
        self.EditExclusionRadius = QtWidgets.QLineEdit("0.0")
        self.SpinAnalysisSamples = QtWidgets.QSpinBox()
        self.SpinAnalysisSamples.setRange(10, 2000)
        self.SpinAnalysisSamples.setValue(200)
        self.ButtonRunAnalysis = QtWidgets.QPushButton("Run Gravity Depth Analysis")
        self.LabelGVec = QtWidgets.QLabel("ĝ = (—)")
        self.LabelGMag = QtWidgets.QLabel("|g| = 0 m/s²")
        playerLayout.addRow("Player X (m):", self.EditPlayerX)
        playerLayout.addRow("Player Y (m):", self.EditPlayerY)
        playerLayout.addRow("Player Z (m):", self.EditPlayerZ)
        playerLayout.addRow("Exclusion radius (m):", self.EditExclusionRadius)
        playerLayout.addRow(self.ButtonSetSurface)
        playerLayout.addRow(self.ButtonRandom, self.ButtonCompute)
        playerLayout.addRow("Analysis samples N:", self.SpinAnalysisSamples)
        playerLayout.addRow(self.ButtonRunAnalysis)
        playerLayout.addRow(self.LabelGVec, self.LabelGMag)
        controlsLayout.addRow(playerGroup)
        layout.addWidget(controls, 0)
        rightWidget = QtWidgets.QWidget()
        rightLayout = QtWidgets.QVBoxLayout()
        rightWidget.setLayout(rightLayout)
        self.Canvas3D = MplCanvas3D(self, width=6, height=4, dpi=100)
        self.CanvasPlot = MplCanvas(self, width=6, height=3, dpi=100)
        rightLayout.addWidget(self.Canvas3D)
        rightLayout.addWidget(self.CanvasPlot)
        layout.addWidget(rightWidget, 1)
    def _ConnectSignals(self):
        self.ButtonGenerate.clicked.connect(self.OnGeneratePlanet)
        self.ButtonCompute.clicked.connect(self.OnComputeGravity)
        self.ButtonSetSurface.clicked.connect(self.OnSetSurface)
        self.ButtonRandom.clicked.connect(self.OnSetRandom)
        self.ButtonRunAnalysis.clicked.connect(self.OnRunDepthAnalysis)
    def _GetPlayerPosition(self):
        try:
            x = float(self.EditPlayerX.text())
            y = float(self.EditPlayerY.text())
            z = float(self.EditPlayerZ.text())
        except:
            return None
        return np.array([x, y, z], dtype=float)

    def _GetExclusionRadius(self):
        try:
            value = float(self.EditExclusionRadius.text())
            return max(0.0, value)
        except:
            return 0.0
    def _Update3DView(self):
        self.Canvas3D.Ax.clear()
        if len(self.Planet.Points) == 0:
            self.Canvas3D.Ax.text(0.5, 0.5, 0.5, "No points")
            self.Canvas3D.draw()
            return
        AllPos = np.array([p.Position for p in self.Planet.Points])
        idx = np.random.choice(AllPos.shape[0], size=min(3000, AllPos.shape[0]), replace=False)
        pts = AllPos[idx]
        self.Canvas3D.Ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c='b', alpha=0.3)
        playerPos = self._GetPlayerPosition()
        if playerPos is not None:
            self.Canvas3D.Ax.scatter([playerPos[0]], [playerPos[1]], [playerPos[2]], s=40, c='r')
        maxRange = np.max(np.linalg.norm(pts, axis=1))
        for axis in [self.Canvas3D.Ax.set_xlim, self.Canvas3D.Ax.set_ylim, self.Canvas3D.Ax.set_zlim]:
            axis(-maxRange, maxRange)
        self.Canvas3D.Ax.set_xlabel("X (m)")
        self.Canvas3D.Ax.set_ylabel("Y (m)")
        self.Canvas3D.Ax.set_zlabel("Z (m)")
        self.Canvas3D.Ax.set_title("3D discrete planet and player position")
        self.Canvas3D.draw()
    def OnGeneratePlanet(self):
        try:
            NMantle = int(self.SpinNMantle.value())
            KCore = int(self.SpinKCore.value())
            MantleMass = float(self.EditMantleMass.text())
            CoreMass = float(self.EditCoreMass.text())
            PlanetRadius = float(self.EditPlanetRadius.text())
            CoreRadius = float(self.EditCoreRadius.text())
        except Exception as e:
            self.LabelStatus.setText("Bad numeric input")
            return
        self.Planet.PlanetRadius = PlanetRadius
        self.Planet.CoreRadius = CoreRadius
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.Planet.Populate(NMantlePoints=NMantle, MantleMassTotal=MantleMass, KCorePoints=KCore, CoreMassTotal=CoreMass, Seed=self.Seed)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.LabelStatus.setText(f"Planet generated: points={len(self.Planet.Points)}")
        self._Update3DView()
    def OnComputeGravity(self):
        playerPos = self._GetPlayerPosition()
        if playerPos is None:
            self.LabelStatus.setText("Bad player coordinates")
            return
        ExclusionRadius = self._GetExclusionRadius()
        gvec = self.Planet.ComputeGravityAt(playerPos, ExclusionRadius=ExclusionRadius)
        gmag = np.linalg.norm(gvec)
        if gmag > np.finfo(float).tiny:
            unit_g = gvec / gmag
            self.LabelGVec.setText(
                f"ĝ = ({unit_g[0]:.8f}, {unit_g[1]:.8f}, {unit_g[2]:.8f})"
            )
        else:
            self.LabelGVec.setText("ĝ = (undefined)")
        self.LabelGMag.setText(f"|g| = {gmag:.8e} m/s²")
        self.LabelStatus.setText("Gravity computed")
        self._Update3DView()
    def OnSetSurface(self):
        r = self.Planet.PlanetRadius
        self.EditPlayerX.setText("0.0")
        self.EditPlayerY.setText("0.0")
        self.EditPlayerZ.setText(str(r))
        self._Update3DView()
    def OnSetRandom(self):
        r = self.Planet.PlanetRadius * (0.5 + 0.5*np.random.rand())
        theta = np.arccos(2*np.random.rand()-1)
        phi = 2*np.pi*np.random.rand()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        self.EditPlayerX.setText(f"{x:.6f}")
        self.EditPlayerY.setText(f"{y:.6f}")
        self.EditPlayerZ.setText(f"{z:.6f}")
        self._Update3DView()
    def OnRunDepthAnalysis(self):
        if len(self.Planet.Points) == 0:
            self.LabelStatus.setText("Generate planet first")
            return
        Samples = int(self.SpinAnalysisSamples.value())
        self._PlotGravityAlongDepth(Samples)
    def _PlotGravityAlongDepth(self, Samples):
        if len(self.Planet.Points) == 0:
            return
        ExclusionRadius = self._GetExclusionRadius()
        Rs = np.linspace(self.Planet.PlanetRadius, 0.0, Samples)
        gvals = []
        for r in Rs:
            pos = np.array([0.0, 0.0, r], dtype=float)
            g = self.Planet.ComputeGravityAt(pos, ExclusionRadius=ExclusionRadius)
            gmag = np.linalg.norm(g)
            if not np.isfinite(gmag):
                gmag = 0.0
            gvals.append(gmag)
        gvals = np.array(gvals)
        DistanceToCenterKm = Rs / 1000.0
        self.CanvasPlot.Ax.clear()
        self.CanvasPlot.Ax.plot(DistanceToCenterKm, gvals)
        self.CanvasPlot.Ax.set_xlabel("Distance to center (km)")
        self.CanvasPlot.Ax.set_ylabel("|g| (m/s²)")
        self.CanvasPlot.Ax.set_title("Gravity magnitude vs distance to center")
        self.CanvasPlot.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GravitySimulatorApp()
    win.resize(1100,700)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
