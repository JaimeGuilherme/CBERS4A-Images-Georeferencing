# 00_clean_intersec.py

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsFeature, QgsField, QgsFields, QgsGeometry, QgsPointXY,
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean, QgsProcessingParameterFeatureSink,
    QgsProcessingParameterDistance, QgsSpatialIndex
)
import math

class PFCCleanIntersectionsAndSplit(QgsProcessingAlgorithm):
    INPUT_POINTS = 'INPUT_POINTS'
    INPUT_LINES = 'INPUT_LINES'
    TOLERANCE = 'TOLERANCE'
    ANG_TOL = 'ANG_TOL'
    DO_SPLIT = 'DO_SPLIT'
    ANG_FLAG_DEG = 'ANG_FLAG_DEG'
    DROP_SAME_WAY = 'DROP_SAME_WAY'
    OUTPUT_POINTS = 'OUTPUT_POINTS'
    OUTPUT_LINES = 'OUTPUT_LINES'

    def tr(self, s): return QCoreApplication.translate('Processing', s)
    def createInstance(self): return PFCCleanIntersectionsAndSplit()
    def name(self): return 'pfc_clean_intersections_and_split'
    def displayName(self): return self.tr('PFC • Limpar Interseções (OSM) + Flags + (Opc.) Quebrar Linhas')
    def group(self): return self.tr('PFC • Redes Viárias')
    def groupId(self): return 'pfc_road_tools'
    def shortHelpString(self):
        return self.tr('Filtra falsos positivos dos pontos de Line Intersections (OSM), '
                       'gera flags de ângulo/continuidade e, opcionalmente, quebra as linhas.')

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_POINTS, self.tr('Camada de pontos (Line Intersections)'),
            [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_LINES, self.tr('Camada de linhas (OSM filtrada)'),
            [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterDistance(
            self.TOLERANCE, self.tr('Tolerância espacial (m)'),
            defaultValue=0.75, parentParameterName=self.INPUT_LINES, minValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.ANG_TOL, self.tr('Tolerância angular para colinearidade (graus)'),
            type=QgsProcessingParameterNumber.Double, defaultValue=12.0, minValue=0.0, maxValue=90.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.ANG_FLAG_DEG, self.tr('Ângulo para flag is_obtuse (graus)'),
            type=QgsProcessingParameterNumber.Double, defaultValue=90.0, minValue=0.0, maxValue=180.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DROP_SAME_WAY, self.tr('Descartar quebras da mesma via (osm_id ou name+ref iguais)'),
            defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DO_SPLIT, self.tr('Quebrar linhas nos pontos válidos'), defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_POINTS, self.tr('Pontos limpos + flags')))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT_LINES, self.tr('Linhas (quebradas ou cópia do original)')))

    # --- helpers geom ---
    @staticmethod
    def _ang_diff(a, b):
        d = abs(a - b) % (2*math.pi)
        return d if d <= math.pi else 2*math.pi - d

    @staticmethod
    def _tangent_angle_linear(g: QgsGeometry, d_along: float):
        L = g.length()
        if not (L > 0): return None
        eps = min(1.0, max(0.005*L, 1e-6))
        d1, d2 = max(0.0, d_along-eps), min(L, d_along+eps)
        p1, p2 = g.interpolate(d1).asPoint(), g.interpolate(d2).asPoint()
        if (p1.x()==p2.x()) and (p1.y()==p2.y()): return None
        return math.atan2(p2.y()-p1.y(), p2.x()-p1.x())

    @staticmethod
    def _same_way(fa, fb):
        # Heurística: igual se osm_id igual OU (name e ref iguais e não vazios)
        def norm(val):
            if val is None: return ''
            return str(val).strip().lower()
        a_osm, b_osm = norm(fa.attribute('osm_id')), norm(fb.attribute('osm_id'))
        if a_osm and b_osm and a_osm == b_osm:
            return True
        an, bn = norm(fa.attribute('name')), norm(fb.attribute('name'))
        ar, br = norm(fa.attribute('ref')),  norm(fb.attribute('ref'))
        if an and bn and an == bn:
            if (ar and br and ar == br) or (not ar and not br):
                return True
        return False

    def processAlgorithm(self, parameters, context, feedback):
        pts_src = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        lines_src = self.parameterAsSource(parameters, self.INPUT_LINES, context)
        if pts_src is None:  raise QgsProcessingException(self.tr('Camada de pontos inválida.'))
        if lines_src is None: raise QgsProcessingException(self.tr('Camada de linhas inválida.'))

        tol = float(self.parameterAsDouble(parameters, self.TOLERANCE, context))
        ang_tol = math.radians(float(self.parameterAsDouble(parameters, self.ANG_TOL, context)))
        ang_flag = float(self.parameterAsDouble(parameters, self.ANG_FLAG_DEG, context))
        drop_same = bool(self.parameterAsBool(parameters, self.DROP_SAME_WAY, context))
        do_split = bool(self.parameterAsBool(parameters, self.DO_SPLIT, context))

        # Saída de pontos: campos originais + flags
        out_fields = QgsFields(pts_src.fields())
        out_fields.append(QgsField('n_lines',  QVariant.Int))
        out_fields.append(QgsField('ang_deg',  QVariant.Double, len=20, prec=3))
        out_fields.append(QgsField('is_obtuse', QVariant.Int))
        out_fields.append(QgsField('same_way',  QVariant.Int))
        pts_sink, pts_dest_id = self.parameterAsSink(
            parameters, self.OUTPUT_POINTS, context,
            out_fields, pts_src.wkbType(), pts_src.sourceCrs())
        if pts_sink is None:
            raise QgsProcessingException(self.tr('Não foi possível criar saída de pontos.'))

        line_sink, line_dest_id = self.parameterAsSink(
            parameters, self.OUTPUT_LINES, context,
            lines_src.fields(), lines_src.wkbType(), lines_src.sourceCrs())
        if line_sink is None:
            raise QgsProcessingException(self.tr('Não foi possível criar saída de linhas.'))

        # Índices
        id2line, line_index = {}, QgsSpatialIndex()
        for lf in lines_src.getFeatures():
            id2line[lf.id()] = lf
            line_index.addFeature(lf)

        pt_index, id2pt = QgsSpatialIndex(), {}
        for pf in pts_src.getFeatures():
            id2pt[pf.id()] = pf
            pt_index.addFeature(pf)

        # Cluster simples de pontos (<= tol/2)
        visited, keep_ids = set(), set()
        half_tol = tol/2.0
        for pid, pf in id2pt.items():
            if pid in visited: continue
            pg = pf.geometry()
            if not pg or pg.isEmpty():
                visited.add(pid); continue
            for nid in pt_index.intersects(pg.buffer(half_tol, 8).boundingBox()):
                nf = id2pt.get(nid)
                if nf and (pg.distance(nf.geometry()) <= half_tol):
                    visited.add(nid)
            keep_ids.add(pid)

        kept_geoms = []  # para split
        # Processa e escreve pontos com flags
        for pid in keep_ids:
            pf = id2pt[pid]
            pg = pf.geometry()
            if not pg or pg.isEmpty(): continue

            # Linhas que realmente tocam o ponto
            touching = []
            for lid in line_index.intersects(pg.buffer(tol, 8).boundingBox()):
                lf = id2line.get(lid)
                if not lf: continue
                lg = lf.geometry()
                if lg and (pg.distance(lg) <= tol):
                    touching.append(lf)

            n = len(touching)
            if n < 2:
                continue

            ang_deg = -1.0
            is_obtuse = 0
            same_way = 0

            # Se exatamente 2 linhas, avalia colinearidade / mesma via / ângulo
            if n == 2:
                l1, l2 = touching[0], touching[1]
                # Heurística "mesma via"
                if self._same_way(l1, l2):
                    same_way = 1
                # Ângulo
                a_list = []
                for lf in (l1, l2):
                    d = lf.geometry().lineLocatePoint(pg)
                    if d < 0: a_list = []; break
                    ang = self._tangent_angle_linear(lf.geometry(), d)
                    if ang is None: a_list = []; break
                    a_list.append(ang)
                if len(a_list) == 2:
                    diff = self._ang_diff(a_list[0], a_list[1])  # [0..pi]
                    # colinear? (0° ou 180° dentro da tolerância)
                    if (diff <= ang_tol) or (abs(diff - math.pi) <= ang_tol):
                        # quebra/continuação: tratar como falso positivo
                        if drop_same or True:
                            # Mesmo que não seja "same_way", continua falso positivo por colinearidade
                            if drop_same:
                                # cai no descarte abaixo
                                pass
                        # marcar angulo para inspeção (ainda que seja descartado)
                    ang_deg = math.degrees(diff)
                    if ang_deg >= ang_flag:
                        is_obtuse = 1

                # Regras de descarte:
                # 1) colinear (já coberta acima pela regra original)
                if len(a_list) == 2:
                    diff = math.radians(ang_deg)
                    if (diff <= ang_tol) or (abs(diff - math.pi) <= ang_tol):
                        continue
                # 2) mesma via (curva/vértice) — opcional
                if drop_same and same_way == 1:
                    continue

            # Se chegou aqui, o ponto é aceito
            kept_geoms.append(pg)

            # Gravar ponto com flags e atributos originais
            newf = QgsFeature(out_fields)
            newf.setAttributes(pf.attributes() + [n, ang_deg, is_obtuse, same_way])
            newf.setGeometry(pg)
            pts_sink.addFeature(newf)

        # (Opcional) Quebrar linhas nos pontos válidos
        if do_split:
            for lf in id2line.values():
                base_geom = QgsGeometry(lf.geometry())
                if not base_geom or base_geom.isEmpty():
                    line_sink.addFeature(lf); continue

                cuts = []
                for pg in kept_geoms:
                    if pg.distance(base_geom) <= tol:
                        d = base_geom.lineLocatePoint(pg)
                        if d >= 0:
                            cuts.append((d, pg.asPoint()))
                if not cuts:
                    line_sink.addFeature(lf); continue

                cuts.sort(key=lambda x: x[0])
                parts = [base_geom]
                for _, pt in cuts:
                    new_parts = []
                    for geom in parts:
                        if pg.distance(geom) <= tol:  # reusa último pg apenas pra checagem rápida
                            status, segs, _ = geom.splitGeometry([QgsPointXY(pt)], False)
                            if status == 0 and segs:
                                new_parts.extend([geom] + segs)
                            else:
                                new_parts.append(geom)
                        else:
                            new_parts.append(geom)
                    parts = new_parts

                for part in parts:
                    if part and not part.isEmpty():
                        nf = QgsFeature(lines_src.fields())
                        nf.setAttributes(lf.attributes())
                        nf.setGeometry(part)
                        line_sink.addFeature(nf)
        else:
            for lf in id2line.values():
                line_sink.addFeature(lf)

        return { self.OUTPUT_POINTS: pts_dest_id, self.OUTPUT_LINES: line_dest_id }
