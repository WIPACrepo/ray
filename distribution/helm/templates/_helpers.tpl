{{- define "cvmfsPersistentVolume" }}
persistentVolumeClaim:
  claimName: cvmfs-volume
{{- end }}
{{- define "cvmfsHostPath" }}
hostPath:
  path: /cvmfs
  type: Directory
{{- end }}
{{- define "prometheusContainer" }}
name: prometheus
image: quay.io/prometheus/prometheus:v2.51.2
imagePullPolicy: IfNotPresent
args:
  - --storage.agent.path="data-agent/"
  - --agent
  - --config.file="/etc/prometheus/config/prometheus.yml"
volumeMounts:
- mountPath: /etc/prometheus/config
  name: config
- mountPath: /tmp/ray
  name: ray-tmp
{{- end }}
{{- define "prometheusConfigVolume" }}
name: config
configMap:
  defaultMode: 420
  name: prometheus-config
{{- }}
