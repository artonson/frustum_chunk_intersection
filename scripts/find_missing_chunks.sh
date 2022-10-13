INPUT_FILENAME=$1

while IFS=' ' read -r scene room type; do
    for chunk in $( seq 0 139 ); do
        filename="/cluster/char/aartemov/spsg/data-geo-color-128/${scene}_room${room}__${type}__${chunk}.sdf"
        if [ -f ${filename} ] ; then
            filename="/cluster/char/aartemov/spsg/output/${scene}_room${room}__${type}__${chunk}.txt"
            if [ ! -f ${filename} ] ; then
                # echo "${filename} present"
            #else
                # echo "${filename} missing"
                echo "${scene} ${room} ${type}"
            fi
        fi

    done
done <"${INPUT_FILENAME:-/dev/stdin}"

