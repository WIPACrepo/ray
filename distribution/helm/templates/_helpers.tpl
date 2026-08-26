{{- define "cvmfsPersistentVolume" }}
persistentVolumeClaim:
  claimName: cvmfs-volume
{{- end }}
{{- define "cvmfsHostPath" }}
hostPath:
  path: /cvmfs
  type: Directory
{{- end }}
