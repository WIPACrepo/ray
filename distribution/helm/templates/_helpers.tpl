{{- define "cvmfsPersistentVolume" }}
persistentVolumeClaim:
  claimName: cvmfs
{{- end }}
{{- define "cvmfsHostPath" }}
hostPath:
  path: /cvmfs
  type: Directory
{{- end }}
