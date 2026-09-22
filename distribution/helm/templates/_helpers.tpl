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
image: quay.io/prometheus/prometheus:latest
imagePullPolicy: IfNotPresent
args:
  - --storage.agent.path=data-agent/
  - --agent
  - --config.file=/etc/prometheus/prometheus.yml
ports:
  - name: web
    protocol: TCP
    containerPort: 9090
resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi
volumeMounts:
- mountPath: /etc/prometheus/prometheus.yml
  subPath: prometheus.yml
  name: config
  readOnly: true
- mountPath: /tmp/ray
  name: ray-tmp
{{- end }}
{{- define "prometheusConfigVolume" }}
name: config
configMap:
  defaultMode: 420
  name: prometheus-config
{{- end }}
