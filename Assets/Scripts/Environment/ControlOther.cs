using UnityEngine;

public class ControlOther : MonoBehaviour
{
    public Rigidbody other;

    void FixedUpdate()
    {
        if (other != null)
        {
            other.velocity = GetComponent<Rigidbody>().velocity;
            var position = GetComponent<Transform>().position;
            var controlledPosition = new Vector3(position.x, position.y + 1.1f, position.z);
            other.GetComponent<Transform>().position = controlledPosition;
        }
    }
}